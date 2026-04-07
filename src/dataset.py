
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transforms import spatial_transform, lowres_transform, intensity_transform
import torch.nn.functional as F
from typing import List, Tuple, Union
from config import Config


class SegmentationDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: str, file_list: List[str], config: Config, augment: bool = False) -> None:
        self.data_dir = data_dir
        self.file_list = file_list
        self.config = config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.file_list[idx]
        
        # 1. Load data (Directly use the normalized data from disk)
        img = np.load(os.path.join(self.data_dir, 'images', name), mmap_mode='r')
        mask = np.load(os.path.join(self.data_dir, 'masks', name), mmap_mode='r')
        img = torch.from_numpy(img.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        # 2. Hardware Alignment (Pad spatial dims to multiples of 16)
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

            for axis in range(1, img.ndim):
                if torch.rand(1) < 0.5:
                    img = torch.flip(img, dims=[axis])
                    mask = torch.flip(mask, dims=[axis])
            

        return img, mask.long()


def get_dataloaders(
    config: Config, 
    data_dir: str = "./data/processed", 
    train_split: float = 0.8, 
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/validation dataloaders following nanochat simplicity principles.
    
    Args:
        config: Configuration object with batch_size attribute
        data_dir: Directory containing ./images/ and ./masks/ subdirectories
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get all .npy files from images directory
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
    train_dataset = SegmentationDataset(data_dir, train_files, config, augment=True)
    val_dataset = SegmentationDataset(data_dir, val_files, config, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader