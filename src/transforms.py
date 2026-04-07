
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def lowres_transform(img: torch.Tensor) -> torch.Tensor:
    if np.random.rand() > 0.25: 
        return img
    zoom = np.random.uniform(0.25, 0.5)
    shape = img.shape[1:]  # ignore channels
    down_shape = [int(s * zoom) for s in shape]

    img = F.interpolate(img.unsqueeze(0), size=down_shape, mode='trilinear', align_corners=False)
    img = F.interpolate(img, size=shape, mode='trilinear', align_corners=False).squeeze(0)
    return img

def intensity_transform(img: torch.Tensor) -> torch.Tensor:
    # 1. Gaussian noise
    if np.random.rand() < 0.15:
        img -= torch.randn_like(img) * np.random.uniform(0, 0.1)  # Additive Gaussian noise
    # 2. Gaussian Blur
    if np.random.rand() < 0.1:
        kernel = 3
        img = F.avg_pool3d(img.unsqueeze(0), kernel, stride=1, padding=1).squeeze(0)
    # 3. Brightness Multiplier
    if np.random.rand() < 0.15:
            img *= np.random.uniform(0.7, 1.3)
    # 4. Gamma transform
    if np.random.rand() < 0.15:
        img = torch.pow(img - img.min(), np.random.uniform(0.7, 1.5)) + img.min()
    return img

def spatial_transform(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if np.random.rand() > 0.2: 
        return img, mask
    
    # 1. Setup params
    rank = img.ndim - 1
    a = np.radians(np.random.uniform(-15, 15)) # ±15 deg
    s = np.random.uniform(0.85, 1.15)
    ca, sa = np.cos(a) * s, np.sin(a) * s

    # 2. Build matrix [1, Rank, Rank+1]
    # Standard 2D rotation/scale matrix, 3D is the same but preserves Z-axis
    if rank == 3:
        mat = [[ca, -sa, 0, 0], 
               [sa,  ca, 0, 0], 
               [0,   0,  s, 0]]
    else:
        mat = [[ca, -sa, 0], 
               [sa,  ca, 0]]
    
    mat = torch.tensor([mat], device=img.device, dtype=torch.float)
    
    # 3. Sample
    grid = F.affine_grid(mat, img[None].shape, align_corners=False)
    
    img  = F.grid_sample(img[None],  grid, mode='bilinear', align_corners=False)[0]
    mask = F.grid_sample(mask[None], grid, mode='nearest',  align_corners=False)[0]
    
    return img, mask