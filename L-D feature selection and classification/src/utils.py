import numpy as np
import torch
import torch.optim as optim
from typing import Callable
import math


class ConstantLearningRate:
    """A learning rate scheduler that uses constant learning rate.
    
    The schedule is a 1-arg callable that produces a learning rate when passed the current step.
    This can be useful for changing the learning rate value across different invocations of optimizer functions.
    
    Args:
        learning_rate: A scalar float. The learning rate.
        name: String. Optional name of the operation. Defaults to 'Constant'.
    """

    def __init__(self, learning_rate, name=None):
        self.learning_rate = learning_rate
        self.name = name or "ConstantLearningRate"

    def __call__(self, step):
        return self.learning_rate

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "name": self.name
        }


class LinearEpochGradualWarmupPolynomialDecayLearningRate:
    """A learning rate scheduler that uses a Linear Epoch Gradual Warmup and Polynomial decay schedule.
    
    It is commonly observed that a linear ramp-up and monotonically decreasing learning rate, whose degree of change
    is carefully chosen, results in a better performing model. This schedule applies a polynomial decay function to an
    optimizer step, given a provided `low_learning_rate`, to reach a `peak_learning_rate` in the given `warmup_steps`,
    and reach a low_learning rate in the remaining steps via a polynomial decay.
    
    Args:
        low_learning_rate: A scalar float. The low learning rate value.
        peak_learning_rate: A scalar float. The peak learning rate value.
        warmup_steps: A scalar int. The number of warmup steps.
        total_steps: A scalar int. The total number of training steps.
        power: A scalar float. The power of the polynomial decay. Defaults to 1.0.
        name: String. Optional name of the operation.
    """

    def __init__(self,
                 low_learning_rate,
                 peak_learning_rate,
                 warmup_steps,
                 total_steps,
                 power=1.0,
                 name=None):
        self.low_learning_rate = low_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.name = name or "LinearEpochGradualWarmupPolynomialDecayLearningRate"
        self.decay_steps = total_steps - warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.low_learning_rate + (self.peak_learning_rate - self.low_learning_rate) * (step / self.warmup_steps)
        else:
            # Polynomial decay
            decay_step = min(step - self.warmup_steps, self.decay_steps)
            return ((self.peak_learning_rate - self.low_learning_rate) *
                    (1 - decay_step / self.decay_steps) ** self.power
                   ) + self.low_learning_rate

    def get_config(self):
        return {
            "low_learning_rate": self.low_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "power": self.power,
            "name": self.name
        }


class ExponentialDecayLearningRate:
    """A learning rate scheduler that uses exponential decay.
    
    Args:
        initial_learning_rate: A scalar float. The initial learning rate.
        decay_steps: A scalar int. The number of steps to decay over.
        decay_rate: A scalar float. The decay rate.
        staircase: Boolean. If True, decay the learning rate at discrete intervals.
        name: String. Optional name of the operation.
    """

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 name=None):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name or "ExponentialDecayLearningRate"

    def __call__(self, step):
        if self.staircase:
            p = step // self.decay_steps
        else:
            p = step / self.decay_steps
        return self.initial_learning_rate * (self.decay_rate ** p)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name
        }


def create_optimizer(model_parameters, optimizer_type='adam', learning_rate=0.001, **kwargs):
    """Create an optimizer for the model.
    
    Args:
        model_parameters: Model parameters to optimize.
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop').
        learning_rate: Learning rate for the optimizer.
        **kwargs: Additional arguments for the optimizer.
    
    Returns:
        PyTorch optimizer instance.
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def apply_learning_rate_schedule(optimizer, lr_schedule, step):
    """Apply learning rate schedule to optimizer.
    
    Args:
        optimizer: PyTorch optimizer.
        lr_schedule: Learning rate schedule function.
        step: Current training step.
    """
    new_lr = lr_schedule(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use for computations.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, filepath):
    """Save a PyTorch model.
    
    Args:
        model: PyTorch model to save.
        filepath: Path to save the model.
    """
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath, device=None):
    """Load a PyTorch model.
    
    Args:
        model: PyTorch model instance.
        filepath: Path to the saved model.
        device: Device to load the model on.
    
    Returns:
        Loaded PyTorch model.
    """
    if device is None:
        device = get_device()
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model