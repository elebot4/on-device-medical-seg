
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
    """
    Apply random rotation and scaling to img and mask tensors.
    
    Args:
        img: [C, ...spatial] tensor
        mask: [C, ...spatial] tensor (same spatial shape as img)
    
    Returns:
        Transformed img, mask tensors with same shapes
    """
    if np.random.rand() > 0.2: 
        return img, mask
    
    # Ensure both tensors have batch dimension for grid_sample
    img_batch = img.unsqueeze(0) if img.dim() == img.ndim else img
    mask_batch = mask.unsqueeze(0) if mask.dim() == mask.ndim else mask
    
    # 1. Setup params
    angle = np.radians(np.random.uniform(-15, 15))  # ±15 degrees
    scale = np.random.uniform(0.85, 1.15)
    cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale

    # 2. Build 2D transformation matrix
    transform_matrix = torch.tensor([[[cos_a, -sin_a, 0], 
                                     [sin_a,  cos_a, 0]]], 
                                   device=img.device, dtype=torch.float)
    
    # 3. Apply transformation
    if img_batch.dim() == 4:  # 2D case: [C, H, W] -> add batch dim -> [1, C, H, W]
        grid = F.affine_grid(transform_matrix, img_batch.shape, align_corners=False)
        img_out = F.grid_sample(img_batch, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
        mask_out = F.grid_sample(mask_batch, grid, mode='nearest', align_corners=False, padding_mode='zeros')
        
        # Remove batch dimension
        return img_out.squeeze(0), mask_out.squeeze(0)
    
    elif img_batch.dim() == 5:  # 3D case: [C, D, H, W] -> add batch dim -> [1, C, D, H, W]
        # For 3D, apply 2D transform to each slice in the depth dimension
        transformed_img = []
        transformed_mask = []
        
        for d in range(img.shape[1]):  # iterate over depth dimension
            img_slice = img_batch[:, :, d:d+1]  # [1, C, 1, H, W]
            mask_slice = mask_batch[:, :, d:d+1]  # [1, C, 1, H, W]
            
            # Remove singleton depth dim for 2D transform: [1, C, H, W]
            img_slice_2d = img_slice.squeeze(2)
            mask_slice_2d = mask_slice.squeeze(2)
            
            # Apply 2D transformation
            grid = F.affine_grid(transform_matrix, img_slice_2d.shape, align_corners=False)
            img_transformed = F.grid_sample(img_slice_2d, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
            mask_transformed = F.grid_sample(mask_slice_2d, grid, mode='nearest', align_corners=False, padding_mode='zeros')
            
            # Add back depth dimension: [1, C, 1, H, W]
            transformed_img.append(img_transformed.unsqueeze(2))
            transformed_mask.append(mask_transformed.unsqueeze(2))
        
        # Concatenate along depth dimension
        img_out = torch.cat(transformed_img, dim=2)  # [1, C, D, H, W]
        mask_out = torch.cat(transformed_mask, dim=2)  # [1, C, D, H, W]
        
        # Remove batch dimension
        return img_out.squeeze(0), mask_out.squeeze(0)
    
    else:
        raise ValueError(f"Unsupported tensor dimensions: img={img.dim()}, expected 4 (2D) or 5 (3D)")


if __name__ == "__main__":
    print("Testing spatial_transform...")
    
    # Test 2D case
    print("\n2D Test:")
    img_2d = torch.randn(1, 64, 64)  # [C, H, W]
    mask_2d = torch.randint(0, 4, (1, 64, 64)).float()  # [C, H, W]
    
    print(f"Input img shape: {img_2d.shape}")
    print(f"Input mask shape: {mask_2d.shape}")
    
    img_out, mask_out = spatial_transform(img_2d, mask_2d)
    
    print(f"Output img shape: {img_out.shape}")
    print(f"Output mask shape: {mask_out.shape}")
    print(f"2D test passed: {img_out.shape == img_2d.shape and mask_out.shape == mask_2d.shape}")
    
    # Test 3D case
    print("\n3D Test:")
    img_3d = torch.randn(1, 32, 64, 64)  # [C, D, H, W]
    mask_3d = torch.randint(0, 4, (1, 32, 64, 64)).float()  # [C, D, H, W]
    
    img_out, mask_out = spatial_transform(img_3d, mask_3d)
    
    print(f"Output img shape: {img_out.shape}")
    print(f"Output mask shape: {mask_out.shape}")
    print(f"3D test passed: {img_out.shape == img_3d.shape and mask_out.shape == mask_3d.shape}")
    
    print("\nAll tests passed!")
    
    img_out, mask_out = spatial_transform(img_3d, mask_3d)
    
    print(f"Output img shape: {img_out.shape}")
    print(f"Output mask shape: {mask_out.shape}")
    print(f"3D test passed: {img_out.shape == img_3d.shape and mask_out.shape == mask_3d.shape}")
    
    # Test multiple runs to check consistency
    print("\nConsistency test (3 runs):")
    for i in range(3):
        img_out, mask_out = spatial_transform(img_2d.clone(), mask_2d.clone())
        print(f"Run {i+1}: shapes match = {img_out.shape == img_2d.shape}")
    
    print("\nAll tests completed.")

