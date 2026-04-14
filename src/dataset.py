
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transforms import intensity_transform, lowres_transform, spatial_transform


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, file_list, augment=False, padding_multiple=16):
        self.data_dir = data_dir
        self.file_list = file_list
        self.augment = augment
        self.padding_multiple = padding_multiple

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        
        # 1. Load data (Directly use the normalized data from disk)
        img = np.load(os.path.join(self.data_dir, 'images', name), mmap_mode='r')
        mask = np.load(os.path.join(self.data_dir, 'masks', name), mmap_mode='r')
        
        print(img.shape)
        print(mask.shape)
        print('-------------------------')
        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        # 2. Pad spatial dimensions to multiples of 16 for efficient convolution operations
        # This ensures compatibility with UNet architectures that use 4 downsampling stages (2^4 = 16)
        # and prevents size mismatches during skip connections and upsampling
        div = 16
        cur = img.shape[1:] # (D, H, W) or (H, W)
        tgt = [((s + div - 1) // div) * div for s in cur]
        if list(cur) != tgt:
            pads = []
            for s, t in zip(reversed(cur), reversed(tgt)):
                pads.extend([(t - s) // 2, (t - s) - (t - s) // 2])
            img = F.pad(img, pads, mode='constant', value=0)
            mask = F.pad(mask, pads, mode='constant', value=0)

        # 4. Apply Augmentations
        if self.augment:
            img, mask = spatial_transform(img, mask)
            img = intensity_transform(img)
            img = lowres_transform(img)

            # Random flips along spatial dimensions
            # img: [C, spatial_dims...], mask: [spatial_dims...]
            spatial_dims = img.ndim - 1  # exclude channel dimension
            for i in range(spatial_dims):
                if torch.rand(1) < 0.5:
                    img = torch.flip(img, dims=[i + 1])    # i+1 to skip channel dim
                    mask = torch.flip(mask, dims=[i])      # no channel dim to skip
            

        return img, mask.long()


def get_dataloaders(data_dir, batch_size, num_stages=4, train_split=0.8, num_workers=4):
    """
    Create train/validation dataloaders following nanoGPT simplicity principles.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size
        num_stages: Number of U-Net stages (used to derive padding multiple)
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Calculate padding multiple based on number of downsampling stages
    padding_multiple = 2 ** num_stages
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
    train_dataset = SegmentationDataset(data_dir, train_files, augment=True, padding_multiple=padding_multiple)
    val_dataset = SegmentationDataset(data_dir, val_files, augment=False, padding_multiple=padding_multiple)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
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
    dataset = SegmentationDataset(data_dir, all_files, augment=False)
    assert len(dataset) == len(all_files), f"Dataset length mismatch: {len(dataset)} vs {len(all_files)}"
    
    # Test 3: Load first sample and check shapes
    img, mask = dataset[0]
    print(f"Sample 0 - Image shape: {img.shape}, Mask shape: {mask.shape}")
    assert img.ndim >= 3, f"Expected at least 3D tensor, got {img.ndim}D"
    assert img.shape[1:] == mask.shape, f"Spatial shape mismatch: {img.shape[1:]} vs {mask.shape}"
    
    # Test 4: Verify padding logic - all spatial dims should be multiples of 16
    spatial_dims = img.shape[1:]
    for i, dim in enumerate(spatial_dims):
        assert dim % 16 == 0, f"Dimension {i} ({dim}) not divisible by 16 after padding"
    print(f"✓ Padding correct: all dimensions divisible by 16")
    
    # Test 5: Check data types
    assert img.dtype == torch.float32, f"Expected float32 image, got {img.dtype}"
    assert mask.dtype == torch.int64, f"Expected int64 mask, got {mask.dtype}"
    print(f"✓ Data types correct: image={img.dtype}, mask={mask.dtype}")
    
    # Test 6: Test augmentation toggle
    dataset_aug = SegmentationDataset(data_dir, all_files[:1], augment=True)  # Single file for determinism
    img_aug, mask_aug = dataset_aug[0]
    print(f"Augmented sample - Image shape: {img_aug.shape}, Mask shape: {mask_aug.shape}")
    assert img_aug.shape[1:] == mask_aug.shape, "Augmented shapes should match"
    
    # Test 7: Dataloader creation with small batch
    try:
        train_loader, val_loader = get_dataloaders(train_split=0.7, num_workers=0)  # No multiprocessing for testing
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