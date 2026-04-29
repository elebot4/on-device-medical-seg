
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transforms import intensity_transform, lowres_transform, spatial_transform


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, file_list, slice_mode='fullres', input_shape=(32, 32, 32), augment=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.slice_mode = slice_mode
        self.input_shape = input_shape
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        
        # 1. Load data using memory mapping (no copy yet)
        img_mmap = np.load(os.path.join(self.data_dir, 'images', name), mmap_mode='r')
        mask_mmap = np.load(os.path.join(self.data_dir, 'masks', name), mmap_mode='r')[None]
        
        # 2. Apply slicing based on mode
        slicer = [slice(None)] * (img_mmap.ndim - 1)

        if self.slice_mode == "sag":
            # Sagittal slice (YZ plane) - slice along X axis (axis 2 for 3D)
            x = np.random.randint(img_mmap.shape[2])
            slicer[2] = x
        elif self.slice_mode == "cor":
            # Coronal slice (XZ plane) - slice along Y axis (axis 1 for 3D)
            y = np.random.randint(img_mmap.shape[1])
            slicer[1] = y
        elif self.slice_mode == "axi":
            # Axial slice (XY plane) - slice along Z axis (axis 0 for 3D)
            z = np.random.randint(img_mmap.shape[0])
            slicer[0] = z
        elif self.slice_mode == "fullres":
            # 3D patch extraction from full resolution volume
            coords = []
            for size, dim in zip(self.input_shape, img_mmap.shape):
                low = size // 2
                high = dim - (size - size // 2) + 1
                if high <= low:
                    # If input_shape is larger than volume dimension, use full dimension
                    coords.append(dim // 2)
                else:
                    coords.append(np.random.randint(low, high))
            
            slicer = []
            for coord, size in zip(coords, self.input_shape):
                start = max(0, coord - size // 2)
                stop = start + size
                slicer.append(slice(start, stop))
        else:
            raise ValueError(f"Unknown slice_mode: {self.slice_mode}")

        # update the slicer to account for channel dimension
        slicer = [slice(None)] + slicer    # Add slice for channel dimension at the beginning
        
        # 3. Extract the slice/patch (this creates a copy)
        img = img_mmap[tuple(slicer)]
        mask = mask_mmap[tuple(slicer)]
        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()
        
        # 5. Resize to exact target shape via padding/cropping
        # This ensures consistent input size for the model
        current_shape = img.shape[1:]  # Skip channel dimension
        target_shape = self.input_shape
        
        if list(current_shape) != list(target_shape):
            # Calculate padding/cropping for each dimension
            pads = []
            crops = []
            
            for curr, tgt in zip(reversed(current_shape), reversed(target_shape)):
                if curr < tgt:
                    # Need padding
                    pad_total = tgt - curr
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    pads.extend([pad_before, pad_after])
                    crops.append(slice(None))
                elif curr > tgt:
                    # Need cropping
                    pads.extend([0, 0])
                    crop_total = curr - tgt
                    crop_start = crop_total // 2
                    crop_end = crop_start + tgt
                    crops.append(slice(crop_start, crop_end))
                else:
                    # Same size
                    pads.extend([0, 0])
                    crops.append(slice(None))
            
            # Apply padding first (if any non-zero pads)
            if any(p > 0 for p in pads):
                img = F.pad(img, pads, mode='constant', value=0)
                mask = F.pad(mask, pads, mode='constant', value=0)
            
            # Apply cropping (reverse order since we reversed above)
            crops = crops[::-1]
            if any(c != slice(None) for c in crops):
                crop_slice = tuple([slice(None)] + crops)  # Keep channel dim
                img = img[crop_slice]
                mask = mask[crop_slice]

        # 6. Apply Augmentations
        if self.augment:
            img, mask = spatial_transform(img, mask)
            img = intensity_transform(img)
            img = lowres_transform(img)

            # Random flips along spatial dimensions
            spatial_dims = img.ndim - 1  # exclude channel dimension
            for i in range(spatial_dims):
                if torch.rand(1) < 0.5:
                    img = torch.flip(img, dims=[i + 1])    # i+1 to skip channel dim
                    mask = torch.flip(mask, dims=[i])      # no channel dim in mask

        # remove singleton channel dim & return 
        mask = mask.squeeze().long() 
        return img, mask


def get_dataloaders(data_dir, batch_size, slice_mode='fullres', input_shape=(32, 32, 32), 
                   train_split=0.8, num_workers=4):
    """
    Create train/validation dataloaders following nanoGPT simplicity principles.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size
        slice_mode: Slicing mode ('axi', 'cor', 'sag', 'fullres')
        input_shape: Target input shape for padding/cropping
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    file_list = [f for f in os.listdir(images_dir) if f.endswith('.npy')]
    if not file_list:
        raise ValueError(f"No .npy files found in {images_dir}")
    
    # Split into train/validation
    split_idx = int(len(file_list) * train_split)
    train_files = file_list[:split_idx]
    val_files = file_list[split_idx:]
    
    # Create datasets
    train_dataset = SegmentationDataset(
        data_dir, train_files, slice_mode=slice_mode, input_shape=input_shape,
        augment=True
    )
    val_dataset = SegmentationDataset(
        data_dir, val_files, slice_mode=slice_mode, input_shape=input_shape,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Basic testing setup - test on 2-3 samples maximum
    import sys
    print("Testing SegmentationDataset...")
    
    # Set global config for testing
    data_dir = "data/processed/Task01_BrainTumour/imagesTr"
    batch_size = 2
    
    # Test 1: Check if data directory exists
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    if not os.path.exists(masks_dir):
        print(f"ERROR: Masks directory not found: {masks_dir}")
        sys.exit(1)
    
    # Get test files (limit to first 3 for fast testing)
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.npy')][:3]
    if len(all_files) < 2:
        print(f"ERROR: Need at least 2 .npy files for testing, found {len(all_files)}")
        sys.exit(1)
    
    print(f"Testing with {len(all_files)} files: {all_files}")
    # Test 2: Dataset initialization and basic loading
    dataset = SegmentationDataset(data_dir, all_files, input_shape=(64, 64, 64), augment=False)
    assert len(dataset) == len(all_files), f"Dataset length mismatch: {len(dataset)} vs {len(all_files)}"
    
    # Test 3: Load first sample and check shapes
    img, mask = dataset[0]
    print(f"Sample 0 - Image shape: {img.shape}, Mask shape: {mask.shape}")
    assert img.ndim >= 3, f"Expected at least 3D tensor, got {img.ndim}D"
    assert img.shape[1:] == mask.shape[1:], f"Spatial shape mismatch: {img.shape[1:]} vs {mask.shape[1:]}"
    
    # Test 4: Verify shape matching - should match input_shape exactly
    spatial_dims = img.shape[1:]
    expected_shape = (64, 64, 64)
    assert spatial_dims == expected_shape, f"Shape mismatch: got {spatial_dims}, expected {expected_shape}"
    print(f"✓ Shape correct: matches target input_shape {expected_shape}")
    
    # Test 5: Check data types
    assert img.dtype == torch.float32, f"Expected float32 image, got {img.dtype}"
    assert mask.dtype == torch.int64, f"Expected int64 mask, got {mask.dtype}"
    print(f"✓ Data types correct: image={img.dtype}, mask={mask.dtype}")
    
    # Test 6: Test augmentation toggle  
    dataset_aug = SegmentationDataset(data_dir, all_files[:1], input_shape=(64, 64, 64), augment=True)  # Single file for determinism
    img_aug, mask_aug = dataset_aug[0]
    print(f"Augmented sample - Image shape: {img_aug.shape}, Mask shape: {mask_aug.shape}")
    assert img_aug.shape[1:] == mask_aug.shape[1:], "Augmented shapes should match"
    
    # Test 7: Dataloader creation with small batch
    try:
        train_loader, val_loader = get_dataloaders(data_dir, batch_size, input_shape=(64, 64, 64), train_split=0.7, num_workers=0)  # No multiprocessing for testing
        print(f"✓ Dataloaders created: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
        
        # Test single batch
        batch_img, batch_mask = next(iter(train_loader))
        print(f"✓ Batch loading: {batch_img.shape}, {batch_mask.shape}")
        assert batch_img.shape[0] <= batch_size, f"Batch size exceeded: {batch_img.shape[0]} > {batch_size}"
        
    except Exception as e:
        print(f"ERROR in dataloader creation: {e}")
        sys.exit(1)
    
    print("✓ All tests passed. Dataset is working correctly.")
    print(f"Image value range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")