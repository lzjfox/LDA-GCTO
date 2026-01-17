import os
import random
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BipartiteGCAROM(nn.Module):
    def __init__(self, dim_lnc: int, dim_dis: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1, skip: bool = True):
        super().__init__()
        self.proj_lnc = nn.Linear(dim_lnc, hidden)
        self.proj_dis = nn.Linear(dim_dis, hidden)
        self.skip = skip
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(GCNConv(hidden, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_lnc: torch.Tensor, x_dis: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        x_l = self.proj_lnc(x_lnc)
        x_d = self.proj_dis(x_dis)
        x = torch.cat([x_l, x_d], dim=0)
        h0 = x
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            if self.skip:
                x = x + h0
        return x  # node embeddings [N_total, hidden]

    @staticmethod
    def score_pairs(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: [E, 2]
        return (z[pairs[:,0]] * z[pairs[:,1]]).sum(dim=-1)  # dot product scores


def load_csv_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = pd.read_csv(os.path.join(data_dir, 'label.csv'), header=None).values  # [N_l, N_d]
    fl = pd.read_csv(os.path.join(data_dir, 'fl.csv'), header=None).values       # [N_l, F_l]
    fd = pd.read_csv(os.path.join(data_dir, 'fd.csv'), header=None).values       # [N_d, F_d]
    assert label.shape[0] == fl.shape[0], f"lncRNA count mismatch: label has {label.shape[0]} rows, fl has {fl.shape[0]}"
    assert label.shape[1] == fd.shape[0], f"disease count mismatch: label has {label.shape[1]} cols, fd has {fd.shape[0]} rows"
    return label.astype(int), fl.astype(np.float32), fd.astype(np.float32)


def build_pos_edges(label: np.ndarray) -> List[Tuple[int, int]]:
    # Return list of (lnc_idx, dis_idx)
    pos = np.argwhere(label == 1)
    return [(int(i), int(j)) for i, j in pos]


def sample_neg_edges(label: np.ndarray, num: int, excluded: set = None) -> List[Tuple[int, int]]:
    N_l, N_d = label.shape
    zeros = np.argwhere(label == 0)
    if excluded is not None:
        zeros = [tuple(map(int, z)) for z in zeros if (int(z[0]), int(z[1])) not in excluded]
    else:
        zeros = [tuple(map(int, z)) for z in zeros]
    if num > len(zeros):
        raise ValueError(f"Not enough negative samples: requested {num}, available {len(zeros)}")
    sel = np.random.choice(len(zeros), size=num, replace=False)
    return [zeros[k] for k in sel]


def make_edge_index_from_pos(pos_edges: List[Tuple[int, int]], N_l: int) -> torch.Tensor:
    # Build undirected edge_index using only positive edges (for message passing graph)
    if len(pos_edges) == 0:
        return torch.empty((2,0), dtype=torch.long)
    src = torch.tensor([u for (u, v) in pos_edges], dtype=torch.long)
    dst = torch.tensor([N_l + v for (u, v) in pos_edges], dtype=torch.long)
    e = torch.stack([src, dst], dim=0)
    e = to_undirected(e)
    return e


def edge_pairs_tensor(edges: List[Tuple[int, int]], N_l: int) -> torch.Tensor:
    if len(edges) == 0:
        return torch.empty((0,2), dtype=torch.long)
    u = torch.tensor([u for (u, v) in edges], dtype=torch.long)
    v = torch.tensor([N_l + v for (u, v) in edges], dtype=torch.long)
    return torch.stack([u, v], dim=1)


def train_one_split(model: BipartiteGCAROM,
                    x_lnc: torch.Tensor,
                    x_dis: torch.Tensor,
                    edge_index_pos_train: torch.Tensor,
                    pos_train_pairs: torch.Tensor,
                    neg_train_pairs: torch.Tensor,
                    pos_val_pairs: torch.Tensor,
                    neg_val_pairs: torch.Tensor,
                    device: torch.device,
                    epochs: int = 200,
                    lr: float = 1e-3,
                    weight_decay: float = 1e-5) -> Tuple[float, dict]:
    model.to(device)
    x_lnc = x_lnc.to(device)
    x_dis = x_dis.to(device)
    edge_index_pos_train = edge_index_pos_train.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_auc = -1.0
    best_state = None
    history = {"train_loss": [], "val_auc": [], "val_aupr": []}

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        z = model(x_lnc, x_dis, edge_index_pos_train)  # [N_total, H]
        pos_scores = model.score_pairs(z, pos_train_pairs.to(device))
        neg_scores = model.score_pairs(z, neg_train_pairs.to(device))
        y = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        logits = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            z = model(x_lnc, x_dis, edge_index_pos_train)
            pos_s = torch.sigmoid(model.score_pairs(z, pos_val_pairs.to(device))).cpu().numpy()
            neg_s = torch.sigmoid(model.score_pairs(z, neg_val_pairs.to(device))).cpu().numpy()
            scores = np.concatenate([pos_s, neg_s])
            labels = np.concatenate([np.ones_like(pos_s), np.zeros_like(neg_s)])
            val_auc = roc_auc_score(labels, scores)
            val_aupr = average_precision_score(labels, scores)
        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(float(val_auc))
        history["val_aupr"].append(float(val_aupr))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_auc, history


def evaluate(model: BipartiteGCAROM,
             x_lnc: torch.Tensor,
             x_dis: torch.Tensor,
             edge_index_pos_train: torch.Tensor,
             pos_test_pairs: torch.Tensor,
             neg_test_pairs: torch.Tensor,
             device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        z = model(x_lnc.to(device), x_dis.to(device), edge_index_pos_train.to(device))
        pos_logits = model.score_pairs(z, pos_test_pairs.to(device))
        neg_logits = model.score_pairs(z, neg_test_pairs.to(device))
        pos_scores = torch.sigmoid(pos_logits).cpu().numpy()
        neg_scores = torch.sigmoid(neg_logits).cpu().numpy()
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        auc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        preds = (scores >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
    return {"AUC": float(auc), "AUPR": float(aupr), "ACC": float(acc), "F1": float(f1)}


def kfold_indices(n: int, k: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds


def run_pipeline(data_dir: str,
                 epochs: int = 200,
                 hidden: int = 64,
                 layers: int = 2,
                 dropout: float = 0.1,
                 kfold: int = 5,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 seed: int = 42,
                 device_str: str = None,
                 out_dir: str = "outputs"):
    set_seed(seed)
    device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label, fl, fd = load_csv_data(data_dir)
    N_l, N_d = label.shape
    dim_l, dim_d = fl.shape[1], fd.shape[1]

    # Build positive edge list
    pos_edges_all = build_pos_edges(label)
    n_pos = len(pos_edges_all)
    print(f"Total positives: {n_pos}")

    # Prepare features tensors
    x_lnc = torch.tensor(fl, dtype=torch.float32)
    x_dis = torch.tensor(fd, dtype=torch.float32)

    folds = kfold_indices(n_pos, kfold, seed)
    metrics_list = []

    for k in range(kfold):
        print(f"\n=== Fold {k+1}/{kfold} ===")
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(kfold) if i != k])
        pos_train = [pos_edges_all[i] for i in train_idx]
        pos_val = [pos_edges_all[i] for i in val_idx]

        edge_index_pos_train = make_edge_index_from_pos(pos_train, N_l)

        neg_train = sample_neg_edges(label, num=len(pos_train), excluded=set(pos_train))
        neg_val = sample_neg_edges(label, num=len(pos_val), excluded=set(pos_train) | set(pos_val))

        pos_train_pairs = edge_pairs_tensor(pos_train, N_l)
        neg_train_pairs = edge_pairs_tensor(neg_train, N_l)
        pos_val_pairs = edge_pairs_tensor(pos_val, N_l)
        neg_val_pairs = edge_pairs_tensor(neg_val, N_l)

        model = BipartiteGCAROM(dim_l, dim_d, hidden=hidden, layers=layers, dropout=dropout, skip=True)
        best_auc, history = train_one_split(model, x_lnc, x_dis, edge_index_pos_train,
                                            pos_train_pairs, neg_train_pairs,
                                            pos_val_pairs, neg_val_pairs,
                                            device, epochs, lr, weight_decay)
        print(f"Best Val AUC: {best_auc:.4f}")

        m = evaluate(model, x_lnc, x_dis, edge_index_pos_train, pos_val_pairs, neg_val_pairs, device)
        print(f"Fold {k+1} Metrics: {m}")
        metrics_list.append(m)

    # Aggregate metrics
    def avg(key):
        return float(np.mean([m[key] for m in metrics_list]))
    summary = {"AUC": avg("AUC"), "AUPR": avg("AUPR"), "ACC": avg("ACC"), "F1": avg("F1")}
    print("\n===== Cross-Validation Summary =====")
    print(summary)

    print("\nTraining final model on full data to produce embeddings...")
    pos_train_full = pos_edges_all
    edge_index_pos_full = make_edge_index_from_pos(pos_train_full, N_l)
    neg_train_full = sample_neg_edges(label, num=len(pos_train_full), excluded=set(pos_train_full))
    model = BipartiteGCAROM(dim_l, dim_d, hidden=hidden, layers=layers, dropout=dropout, skip=True).to(device)

    x_lnc_d = x_lnc.to(device)
    x_dis_d = x_dis.to(device)
    edge_index_d = edge_index_pos_full.to(device)
    pos_pairs = edge_pairs_tensor(pos_train_full, N_l).to(device)
    neg_pairs = edge_pairs_tensor(neg_train_full, N_l).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        z = model(x_lnc_d, x_dis_d, edge_index_d)
        pos_scores = model.score_pairs(z, pos_pairs)
        neg_scores = model.score_pairs(z, neg_pairs)
        y = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        logits = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print(f"Final-train epoch {epoch}/{epochs} loss={loss.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Generate embeddings
    model.eval()
    with torch.no_grad():
        z = model(x_lnc_d, x_dis_d, edge_index_d).cpu().numpy()
    z_lnc = z[:N_l]
    z_dis = z[N_l:]

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(z_lnc).to_csv(os.path.join(out_dir, 'lncRNA_embeddings.csv'), header=False, index=False)
    pd.DataFrame(z_dis).to_csv(os.path.join(out_dir, 'disease_embeddings.csv'), header=False, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCA-ROM-based bipartite link prediction for lncRNA-disease')
    parser.add_argument('--data_dir', type=str, default='data1', help='Directory containing label.csv, fl.csv, fd.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None, help="cuda or cpu; default auto")
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    run_pipeline(data_dir=args.data_dir,
                 epochs=args.epochs,
                 hidden=args.hidden,
                 layers=args.layers,
                 dropout=args.dropout,
                 kfold=args.kfold,
                 lr=args.lr,
                 weight_decay=args.weight_decay,
                 seed=args.seed,
                 device_str=args.device,
                 out_dir=args.out_dir)