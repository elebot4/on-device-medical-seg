import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
import time
from typing import Tuple

from config import Config
from model import UNet
from dataset import SegmentationDataset, get_dataloaders
from loss import dice_loss

def train() -> None:
    # 1. Setup
    cfg = Config()
    device = torch.device(cfg.device)
    torch.manual_seed(42)
    
    # 2. Data
    train_loader, val_loader = get_dataloaders(cfg)
    
    # 3. Model, Optimizer, Scaler
    model = UNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    scaler = GradScaler() # For Mixed Precision
    
    print(f"Starting training on {device}...")

    for epoch in range(cfg.nb_epochs):
        model.train()
        t0 = time.time()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True) # Index tensor [B, D, H, W]

            # Mixed Precision Forward Pass
            with autocast():
                preds = model(images) # List of [Res1, Res2, ...]
                
                total_loss = 0
                # Deep Supervision loop using strided slicing
                for i, p in enumerate(preds):
                    stride = 2**i
                    # Strided slice: [B, D, H, W] -> [B, D/s, H/s, W/s]
                    # This is O(1) memory/time overhead
                    t_idx = masks[:, ::stride, ::stride, ::stride] if stride > 1 else masks
                    
                    # GPU-side one-hotting for Dice
                    t_onehot = F.one_hot(t_idx, num_classes=cfg.out_channels)
                    t_onehot = t_onehot.permute(0, 4, 1, 2, 3).float()

                    # Loss Calculation
                    loss_ce = F.cross_entropy(p, t_idx)
                    loss_dice = dice_loss(F.softmax(p, dim=1), t_onehot)
                    
                    weight = 1 / (2**i)
                    total_loss += weight * (loss_ce + loss_dice)
                
                # Normalize by sum of weights
                total_loss /= sum([1/(2**i) for i in range(len(preds))])

            # Backward Pass with Scaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # 4. Validation 
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                # In eval mode, UNet returns only the high-res tensor
                out = model(images) 
                
                # Simple Dice Metric for monitoring
                p = F.one_hot(out.argmax(1), num_classes=cfg.out_channels).permute(0, 4, 1, 2, 3).float()
                t = F.one_hot(masks, num_classes=cfg.out_channels).permute(0, 4, 1, 2, 3).float()
                val_dice += (1 - dice_loss(p, t)) # 1 - (1-dice) = dice

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.nb_epochs} | Loss: {total_loss.item():.4f} | Val Dice: {val_dice/len(val_loader):.4f} | Time: {dt:.2f}s")

        # 5. Save Checkpoint 
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg,
            }
            torch.save(checkpoint, f"ckpt_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()