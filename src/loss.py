"""
Functions and classes for losses and metrics related to evaluation of segmentation models performance for optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Stateless dice loss function for 2d & 3d segmentation.
    """
    
    p = F.softmax(logits, dim = 1)
    reduce_axis = tuple(range(2, p.ndim)) # exclude batch & channel dimensions
    t = targets

    intersection = (p * targets).sum(dim = reduce_axis)
    p = p.sum(dim = reduce_axis)
    t = targets.sum(dim = reduce_axis)
    d = 2 * (intersection + smooth) / (p + t + smooth)

    # we remove locations where t is background
    return -d[t > 0].mean()

FUNCTIONAL_LOSSES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "CrossEntropy": F.binary_cross_entropy_with_logits,
    "Dice": dice_loss,
    "MSE": F.mse_loss,
    "L1": F.l1_loss
}

def get_loss_fn(args) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Loss function factory to create composite loss functions. Loss functions is created from user defined argument string
    e.g "0.5*CrossEntropy+0.5*Dice"
    """
    
    loss_parts = []
    for loss in args.loss.split("+"):
        weight, loss_type = loss.split("*")
        loss_parts.append((float(weight), FUNCTIONAL_LOSSES[loss_type]))
        
    # define the composite loss function to return
    def loss_fn(logits, targets):
        return sum(w * fn(logits, targets) for w, fn in loss_parts)
    return loss_fn

 

if __name__ == "__main__":
    # test the loss functions with dummy data
    class Args:
        loss = "0.5*CrossEntropy+0.5*Dice"
    
    args = Args()
    loss_fn = get_loss_fn(args)
    
    logits = torch.randn(2, 2, 4, 4)
    targets = torch.randint(0, 2, (2, 2, 4, 4)).float()
    
    loss_value = loss_fn(logits, targets)

    print("Loss value:", loss_value.item())

