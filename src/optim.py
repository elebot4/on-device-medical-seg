import torch
import torch.nn as nn
from typing import Union

from config import Config

def get_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    # --- Step 1: Weight Decay Decoupling ---
    # We separate parameters into those that get decay and those that don't (biases/norms)
    decay = set()
    no_decay = set()
    whitelist = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    blacklist = (nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # --- Step 2: Optimizer Selection ---
    if config.optimizer == "ADAMw":
        return torch.optim.AdamW(optim_groups, lr=config.lr, betas=config.betas)
    elif config.optimizer == "ADAM":
        return torch.optim.Adam(optim_groups, lr=config.lr, betas=config.betas)
    elif config.optimizer == "SGD":
        return torch.optim.SGD(optim_groups, lr=config.lr, momentum=config.momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

def get_scheduler(
    optimizer: torch.optim.Optimizer, 
    config: Config, 
    steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    # Total iterations is usually better for schedulers than "epochs"
    total_steps = config.nb_epochs * steps_per_epoch

    if config.scheduler == "PolyLR":
        # Standard for medical: (1 - step/max_step)^gamma
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: (1 - s / total_steps) ** config.gamma
        )
    
    elif config.scheduler == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr * 10, total_steps=total_steps
        )
    
    elif config.scheduler == "MultiStepLR":
        # Turn percentages (0.5, 0.75) into absolute step counts
        milestones = [int(m * total_steps) for m in config.milestones]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Fallback to Constant
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)