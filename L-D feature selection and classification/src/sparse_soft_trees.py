import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import math

class ProximalGroupL0:
    """Applies group-wise proximal operator after gradient update for Group L0.
    
    Solves the following minimization problem:
        min_Z (1/(2*lr))*||Z-W_{t}))||_2^2 + lam_lo * sum_{j=1}^p ||Z_j||_0
        s.t. W_{t} = W_{t-1}-lr*Df(W_{t-1})
    
    Proximal Operator:
    For each group w_t, the proximal operator is a soft-thresholding operator:
        H_{lr*lam_l0, lr*lam_l2}(w_t) = {w_t  if ||w_t||_2 >= sqrt(2*lr*lam_l0),
                                        {0    o.w. 

    References:
        - End-to-end Feature Selection Approach for Learning Skinny Trees
          [https://arxiv.org/pdf/2310.18542.pdf]
        - Grouped Variable Selection with Discrete Optimization: Computational and Statistical Perspectives
          [https://arxiv.org/pdf/2104.07084.pdf]
        

    Inputs:
        w: Float Tensor of shape (num_features, num_trees*num_nodes).
        
    Returns:
        w: Float Tensor of shape (num_features, num_trees*num_nodes).
    """
    def __init__(self, lr=0.01, lam=0., use_annealing=False, temperature=0.1, name='ProximalGroupL0'):
        self.lam = lam
        self.lr = lr if not callable(lr) else lr
        self.use_annealing = use_annealing
        self.name = name
        self.iterations = 0
        if self.use_annealing:
            self.temperature = temperature
    
    def __call__(self, w):
        self.iterations += 1
        lr_val = self.lr(self.iterations) if callable(self.lr) else self.lr
        lam_lr = self.lam * lr_val
        
        if self.use_annealing:
            scheduler = (1.0 - torch.exp(
                -self.temperature * self.iterations
            ))
        else:
            scheduler = 1.0
        
        # The proximity operator for the group l0 is hard thresholding on each group.
        w_norm = torch.norm(w, p=2, dim=1, keepdim=True)
        hard_threshold = torch.sqrt(
            torch.tensor(2.0 * lam_lr * scheduler, device=w.device, dtype=w.dtype)
        )
        
        mask = w_norm > hard_threshold
        w = torch.where(mask, w, torch.zeros_like(w))
        return w

# Utility functions to count active features in a Tree Ensemble
def count_selected_weights(model):
    """Count selected weights in the model."""
    weights = None
    for module in model.modules():
        if isinstance(module, TreeEnsembleWithGroupSparsity) and hasattr(module, 'dense_layer'):
            weights = module.dense_layer.weight.data
            break
    
    if weights is not None:
        return torch.sum(torch.mean((torch.abs(weights) > 0.0).float(), dim=1)).item()
    return 0

def count_approximately_selected_weights(model):
    """Count approximately selected weights in the model."""
    weights = None
    for module in model.modules():
        if isinstance(module, TreeEnsembleWithGroupSparsity) and hasattr(module, 'dense_layer'):
            weights = module.dense_layer.weight.data
            break
    
    if weights is not None:
        return torch.sum(torch.mean((torch.abs(weights) > 1e-4).float(), dim=1)).item()
    return 0

def count_selected_features(model):
    """Count selected features in the model."""
    weights = None
    for module in model.modules():
        if isinstance(module, TreeEnsembleWithGroupSparsity) and hasattr(module, 'dense_layer'):
            weights = module.dense_layer.weight.data
            break
    
    if weights is not None:
        return torch.sum((torch.norm(weights, p=2, dim=0) > 0.0).float()).item()
    return 0

def count_approximately_selected_features(model):
    """Count approximately selected features in the model."""
    weights = None
    for module in model.modules():
        if isinstance(module, TreeEnsembleWithGroupSparsity) and hasattr(module, 'dense_layer'):
            weights = module.dense_layer.weight.data
            break
    
    if weights is not None:
        return torch.sum(((1/math.sqrt(weights.shape[0])) * torch.norm(weights, p=2, dim=0) > 1e-4).float()).item()
    return 0


class SparsityHistory:
    """Callback class to save training loss and the number of features."""
    def __init__(self):
        self.losses = []
        self.selected_features = []
        self.approximately_selected_features = []
        self.selected_weights = []
        self.approximately_selected_weights = []
    
    def on_train_begin(self, model):
        self.selected_features = [count_selected_features(model)]
        self.approximately_selected_features = [count_approximately_selected_features(model)]
        self.selected_weights = [count_selected_weights(model)]
        self.approximately_selected_weights = [count_approximately_selected_weights(model)]
    
    def on_epoch_end(self, model, loss):
        self.losses.append(loss)
        self.selected_features.append(count_selected_features(model))
        self.approximately_selected_features.append(count_approximately_selected_features(model))
        self.selected_weights.append(count_selected_weights(model))
        self.approximately_selected_weights.append(count_approximately_selected_weights(model))


class TreeEnsembleWithGroupSparsity(nn.Module):
    """An ensemble of soft decision trees.
    
    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.
    
    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with 
        num_trees units, each corresponding to the hyperplane of one tree.
    
    Input:
        An input tensor of shape = (batch_size, ...)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(self,
                 num_trees,
                 max_depth,
                 leaf_dims,
                 activation='sigmoid',
                 node_index=0,
                 internal_eps=0,
                 kernel_regularizer=None,
                 kernel_constraint=None):
        super(TreeEnsembleWithGroupSparsity, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.node_index = node_index
        self.leaf = node_index >= 2**max_depth - 1
        self.max_split_nodes = 2**max_depth - 1
        self.internal_eps = internal_eps
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = getattr(F, activation, torch.sigmoid)
        
        if not self.leaf:
            if self.node_index == 0:
                self.dense_layer = None  # Will be set in build method
            
            # Create masking tensor
            masking = torch.zeros(1, self.num_trees, self.max_split_nodes)
            masking[:, :, self.node_index] = 1
            self.register_buffer('masking', masking)
            
            self.left_child = TreeEnsembleWithGroupSparsity(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+1,
            )
            self.right_child = TreeEnsembleWithGroupSparsity(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+2,
            )
        else:
            # Leaf node weight
            self.leaf_weight = nn.Parameter(
                torch.randn(1, self.leaf_dims, self.num_trees)
            )
    
    def build(self, input_shape):
        """Build the dense layer for the root node."""
        if self.node_index == 0 and not self.leaf:
            input_dim = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape
            self.dense_layer = nn.Linear(input_dim, self.num_trees * self.max_split_nodes)
    
    def forward(self, input_tensor, prob=None):
        if prob is None:
            prob = torch.ones(input_tensor.shape[0], self.num_trees, device=input_tensor.device)
        
        if self.node_index == 0:
            if self.dense_layer is None:
                self.build(input_tensor.shape)
            
            output = self.dense_layer(input_tensor)
            output = output.view(output.shape[0], self.num_trees, self.max_split_nodes)
            
            # Apply group L0 regularization if constraint is provided
            if isinstance(self.kernel_constraint, ProximalGroupL0):
                w = self.dense_layer.weight
                w_norm = torch.norm(w, p=2, dim=1)
                lam = self.kernel_constraint.lam
                
                if self.kernel_constraint.use_annealing:
                    lam = lam * (1.0 - torch.exp(
                        -self.kernel_constraint.temperature * self.kernel_constraint.iterations
                    ))
                
                nnz = torch.sum((w_norm > 0).float())
                regularization = lam * nnz
                # Note: In PyTorch, regularization is typically handled in the loss function
        else:
            output = input_tensor
        
        if not self.leaf:
            # shape = (batch_size, num_trees)
            current_prob = torch.clamp(
                self.activation(
                    torch.sum(output * self.masking, dim=-1)
                ),
                self.internal_eps,
                1 - self.internal_eps
            )
            
            left_output = self.left_child(output, current_prob * prob)
            right_output = self.right_child(output, (1 - current_prob) * prob)
            return left_output + right_output
        else:
            # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
            # prob's shape = (batch_size, num_trees)
            return torch.sum(prob.unsqueeze(1) * self.leaf_weight, dim=2)