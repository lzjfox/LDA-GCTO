"""Creates models for multitask learning with Neural Networks using PyTorch.
"""
import torch
import torch.nn as nn
from .sparse_soft_trees import TreeEnsembleWithGroupSparsity


def create_model(
    input_dim,
    num_trees,
    depth,
    leaf_dims,
    activation='sigmoid',
    kernel_regularizer=None,
    kernel_constraint=None,
):
    """Creates a submodel for a task with soft decision trees.
    
    Args:
      input_dim: Input dimension, int scalar.
      num_trees: Number of trees in the ensemble, int scalar.
      depth: Depth of each tree. Note: in the current implementation,
        all trees are fully grown to depth, int scalar.
      leaf_dims: list of dimensions of leaf outputs,
        int tuple of shape (num_layers, ).
      activation: 'sigmoid'
      kernel_regularizer: Regularizer for the kernel weights.
      kernel_constraint: Constraint for the kernel weights.
      
    Returns:
      PyTorch model instantiation
    """
    
    class SkinnyTreesModel(nn.Module):
        def __init__(self):
            super(SkinnyTreesModel, self).__init__()
            self.tree_ensemble = TreeEnsembleWithGroupSparsity(
                num_trees,
                depth,
                leaf_dims[-1] if isinstance(leaf_dims, (list, tuple)) else leaf_dims,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_constraint=kernel_constraint
            )
            # Build the dense layer
            self.tree_ensemble.build(input_dim)
        
        def forward(self, x):
            return self.tree_ensemble(x)
            
        def get_regularization_loss(self):
            loss = 0.0
            if hasattr(self.tree_ensemble, 'kernel_regularizer') and self.tree_ensemble.kernel_regularizer is not None:
                if hasattr(self.tree_ensemble, 'dense_layer') and self.tree_ensemble.dense_layer is not None:
                    loss += self.tree_ensemble.kernel_regularizer(self.tree_ensemble.dense_layer.weight)
            return loss
    
    model = SkinnyTreesModel()
    return model