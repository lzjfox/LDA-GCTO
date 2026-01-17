from CV import kfold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
from typing import  Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.models import create_model
from src.sparse_soft_trees import ProximalGroupL0, SparsityHistory
from src.utils import (
    ConstantLearningRate,
    LinearEpochGradualWarmupPolynomialDecayLearningRate,
    get_device, set_seed, count_parameters
)

import warnings
warnings.filterwarnings("ignore")


def train_and_evaluate_fold(X_train: torch.Tensor, y_train: torch.Tensor,
                            X_val: torch.Tensor, input_dim: int, device: torch.device) -> Dict:

    model = create_model(
        input_dim=input_dim,
        num_trees=80,
        depth=6,
        leaf_dims=1,
        kernel_constraint=ProximalGroupL0(
            lr=ConstantLearningRate(0.001),
            lam=0.01
        )
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)

    model.train()
    num_epochs = 300

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)

        main_loss = criterion(outputs.squeeze(), y_train)

        total_loss = main_loss
        total_loss.backward()
        optimizer.step()

        if hasattr(model.tree_ensemble, 'kernel_constraint') and model.tree_ensemble.kernel_constraint:
            if hasattr(model.tree_ensemble, 'dense_layer') and model.tree_ensemble.dense_layer is not None:
                with torch.no_grad():
                    w = model.tree_ensemble.dense_layer.weight.data            # [out_features, in_features]
                    w_t = model.tree_ensemble.kernel_constraint(w.t())         # (num_features, num_trees*num_nodes)
                    model.tree_ensemble.dense_layer.weight.data = w_t.t()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs = torch.sigmoid(val_outputs.squeeze())

    return val_probs



time = 20
k = 5
for mm in range(1, 4):
    for cv in range(1,5):
        data_file = '../data' + str(mm) + '/data256.csv'
        label_file = '../data' + str(mm) + '/label.csv'
        data = pd.read_csv(data_file, header=None, index_col=None).to_numpy()
        label = pd.read_csv(label_file, index_col=None, header=None).to_numpy()
        label_copy = label.copy()
        row, col = label.shape
        if cv == 4:
            c = np.array([(i, j) for i in range(row) for j in range(col)])
        else:
            a = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j]])
            b = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])
            np.random.shuffle(b)
            sample = len(a)
            b = b[:sample]
        mPREs = np.array([])
        mACCs = np.array([])
        mRECs = np.array([])
        mAUCs = np.array([])
        mAUPRs = np.array([])
        mF1 = np.array([])

        for j in range(time):
            if cv == 4:
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            elif cv == 3:
                a_tr, a_te = np.array(kfold(a, k=k, row=row, col=col, cv=cv))
                b_tr, b_te = np.array(kfold(b, k=k, row=row, col=col, cv=cv))
            else:
                c = np.vstack([a, b])
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            for  i in range(k):
                if cv == 4:
                    b_tr = []
                    a_tr = []
                    # print(c_tr[i])
                    for ep in c_tr[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_tr.append(c[ep])
                        else:
                            a_tr.append(c[ep])
                    b_te = []
                    a_te = []
                    for ep in c_te[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_te.append(c[ep])
                        else:
                            a_te.append(c[ep])
                    b_te = np.array(b_te)
                    b_tr = np.array(b_tr)
                    a_te = np.array(a_te)
                    a_tr = np.array(a_tr)
                    np.random.shuffle(b_te)
                    np.random.shuffle(a_te)
                    np.random.shuffle(b_tr)
                    np.random.shuffle(a_tr)
                    a_tr = a_tr[:len(b_tr)]
                    a_te = a_te[:len(b_te)]
                    train_sample = np.vstack([a_tr, b_tr])
                    test_sample = np.vstack([a_te, b_te])
                elif cv == 3:
                    train_sample = np.vstack([np.array(a[a_tr[i]]), np.array(b[b_tr[i]])])
                    test_sample = np.vstack([np.array(a[a_te[i]]), np.array(b[b_te[i]])])
                else:
                    train_sample = np.array(c[c_tr[i]])
                    test_sample = np.array(c[c_te[i]])
                train_land = train_sample[:, 0] * col + train_sample[:, 1]
                test_land = test_sample[:, 0] * col + test_sample[:, 1]

                X_tr = data[train_land][:, :-1]
                y_tr = data[train_land][:, -1]
                X_te = data[test_land][:, :-1]
                y_te = data[test_land][:, -1]

                device = get_device()

                X_train = torch.FloatTensor(X_tr)
                y_train = torch.FloatTensor(y_tr)
                X_val = torch.FloatTensor(X_te)
                score = train_and_evaluate_fold(X_train, y_train, X_val, X_tr.shape[1], device).cpu().numpy()

                pred = []
                prob =  score
                for n in prob:
                    if n > 0.5:
                        pred.append(1)
                    else:
                        pred.append(0)
                pre_label = np.array(pred)

                fpr, tpr, threshold = roc_curve(y_te, score)
                pre, rec_, _ = precision_recall_curve(y_te, score)
                acc = accuracy_score(y_te, pre_label)
                rec = recall_score(y_te, pre_label)
                f1 = f1_score(y_te, pre_label)
                Pre = precision_score(y_te, pre_label)
                au = auc(fpr, tpr)
                apr = auc(rec_, pre)
                mPREs = np.append(mPREs, Pre)
                mACCs = np.append(mACCs, acc)
                mRECs = np.append(mRECs, rec)
                mAUCs = np.append(mAUCs, au)
                mAUPRs = np.append(mAUPRs, apr)
                mF1 = np.append(mF1, f1)

                curve_1 = np.vstack([fpr, tpr])
                curve_1 = pd.DataFrame(curve_1.T)
                curve_1.to_csv('./co_s/d' + str(mm) + 'c' + str(cv) + '_c' + str(au) + '.csv',
                               header=None, index=None)

                curve_2 = np.vstack([rec_, pre])
                curve_2 = pd.DataFrame(curve_2.T)
                curve_2.to_csv('./co_s/d' + str(mm) + 'c' + str(cv) + '_r' + str(apr) + '.csv',
                               header=None, index=None)

                print('Precision is :{}'.format(Pre))
                print('Recall is :{}'.format(rec))
                print("ACC is: {}".format(acc))
                print("F1 is: {}".format(f1))
                print("AUC is: {}".format(au))
                print('AUPR is :{}'.format(apr))

            print("One time 5-Folds computed.")

        toa = np.vstack([mAUCs, mAUPRs])
        toa = pd.DataFrame(toa)
        toa.to_csv('res_data/' + 'cv' + str(cv) + '_data' + str(mm) + '.csv', header=None, index=None)
        PRE = mPREs.mean()
        ACC = mACCs.mean()
        REC = mRECs.mean()
        AUC = mAUCs.mean()
        AUPR = mAUPRs.mean()
        F1 = mF1.mean()

        PRE_err = np.std(mPREs)
        ACC_err = np.std(mACCs)
        REC_err = np.std(mRECs)
        AUC_err = np.std(mAUCs)
        AUPR_err = np.std(mAUPRs)
        F1_err = np.std(mF1)

        print('\n')
        print("PRE is:{}±{}".format(round(PRE, 4), round(PRE_err, 4)))
        print("REC is:{}±{}".format(round(REC, 4), round(REC_err, 4)))
        print("ACC is:{}±{}".format(round(ACC, 4), round(ACC_err, 4)))
        print("F1 is:{}±{}".format(round(F1, 4), round(F1_err, 4)))
        print('AUC is :{}±{}'.format(round(AUC, 4), round(AUC_err, 4)))
        print('AUPR is :{}±{}'.format(round(AUPR, 4), round(AUPR_err, 4)))

        f_1 = open('./co_s/' + 'res_cv.txt', 'a')
        f_1.write('data' + str(mm) + 'cv' + str(cv) + ':\n')
        f_1.write(str(round(PRE, 4)) + '±' + str(round(PRE_err, 4)) + '\t')
        f_1.write(str(round(REC, 4)) + '±' + str(round(REC_err, 4)) + '\t')
        f_1.write(str(round(ACC, 4)) + '±' + str(round(ACC_err, 4)) + '\t')
        f_1.write(str(round(F1, 4)) + '±' + str(round(F1_err, 4)) + '\t')
        f_1.write(str(round(AUC, 4)) + '±' + str(round(AUC_err, 4)) + '\t')
        f_1.write(str(round(AUPR, 4)) + '±' + str(round(AUPR_err, 4)) + '\t\n\n')
        f_1.close()
        print('\nwrite\n')