
import numpy as np
import torch
import torch.nn.functional as F


def lowres_transform(img):
    if np.random.rand() > 0.25: 
        return img
    zoom = np.random.uniform(0.25, 0.5)
    shape = img.shape[1:]  # ignore channels
    down_shape = [int(s * zoom) for s in shape]

    mode = 'trilinear' if len(shape) == 3 else 'bilinear'
    img = F.interpolate(img.unsqueeze(0), size=down_shape, mode=mode, align_corners=False)
    img = F.interpolate(img, size=shape, mode=mode, align_corners=False).squeeze(0)
    return img

def intensity_transform(img):
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

def spatial_transform(img, mask):
    if np.random.rand() > 0.2: 
        return img, mask
    
    # 1. Setup params
    rank = img.ndim - 1
    a = np.radians(np.random.uniform(-15, 15)) # ±15 deg
    s = np.random.uniform(0.85, 1.15)
    ca, sa = np.cos(a) * s, np.sin(a) * s

    # 2. Build matrix [1, Rank, Rank+1] 
    if rank == 3:
        # For 3D: rotation in XY plane, uniform scale in Z
        mat = [[ca, -sa, 0, 0], 
               [sa,  ca, 0, 0], 
               [0,   0,  s, 0]]
        mat = torch.tensor([mat], device=img.device, dtype=torch.float)
        grid = F.affine_grid(mat, img[None].shape, align_corners=False)
        
        # 3D sampling
        img  = F.grid_sample(img[None], grid, mode='trilinear', align_corners=False)[0]
        mask = F.grid_sample(mask[None], grid, mode='nearest',  align_corners=False)[0]
    else:
        # For 2D
        mat = [[ca, -sa, 0], 
               [sa,  ca, 0]]
        mat = torch.tensor([mat], device=img.device, dtype=torch.float)
        grid = F.affine_grid(mat, img[None].shape, align_corners=False)
        
        # 2D sampling  
        img  = F.grid_sample(img[None], grid, mode='bilinear', align_corners=False)[0]
        mask = F.grid_sample(mask[None], grid, mode='nearest',  align_corners=False)[0]
    
    return img, mask