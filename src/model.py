import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Union, Callable

from config import Config


# -----------------------------------------------------------------------------
# Building Blocks

class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, block_config: Config) -> None:
        super().__init__()
        
        # 1. Local aliasing and configuration of operators based on spatial dimensions
        conv = nn.Conv3d if len(block_config.input_shape) == 3 else nn.Conv2d
        dropout = nn.Dropout3d if len(block_config.input_shape) == 3 else nn.Dropout2d
        
        if block_config.norm_type == 'group':
            norm = lambda c: nn.GroupNorm(8, c)
        elif block_config.norm_type == 'batch':
            norm = nn.BatchNorm3d if len(block_config.input_shape) == 3 else nn.BatchNorm2d
        elif block_config.norm_type == 'instance':
            norm = nn.InstanceNorm3d if len(block_config.input_shape) == 3 else nn.InstanceNorm2d
        else:
            norm = nn.Identity

        if block_config.act_type == 'relu':
            act = nn.ReLU
        elif block_config.act_type == 'gelu':
            act = nn.GELU
        else:
            act = lambda: nn.LeakyReLU(0.01)

        # 2. Architectural definition
        self.shortcut = conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        self.net = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch),
            act(),
            dropout(block_config.dropout),
            conv(out_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch),
            act(),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------------------------------------------------------
# Main Model

class UNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # 1. Local aliasing and configuration of operators based on spatial dimensions
        conv = nn.Conv3d if len(config.input_shape) == 3 else nn.Conv2d
        conv_t = nn.ConvTranspose3d if len(config.input_shape) == 3 else nn.ConvTranspose2d
        
        # Channel schedule: e.g., [32, 64, 128, 256]
        chs = [config.base_chs * (2**i) for i in range(config.num_stages)]
        
        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        curr_in = config.in_channels
        for i in range(config.num_stages - 1):
            self.encoders.append(Block(curr_in, chs[i], config))
            self.downs.append(conv(chs[i], chs[i], kernel_size=2, stride=2))
            curr_in = chs[i]

        # --- Bottleneck ---
        self.bottleneck = Block(chs[-2], chs[-1], config)
        
        # --- Decoder & Deep Supervision Heads ---
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList() # One head for each decoder stage for deepsupervision

        # We build the decoder from deepest to shallowest
        for i in reversed(range(config.num_stages - 1)):
            self.ups.append(conv_t(chs[i+1], chs[i], kernel_size=2, stride=2))
            self.decoders.append(Block(chs[i] * 2, chs[i], config))
            self.heads.append(conv(chs[i], config.out_channels, kernel_size=1))

        self.apply(self._init_weights)
        print(f"UNet initialized: {sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def _init_weights(self, m: nn.Module) -> None:
        conv_layers = (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)
        norm_layers = (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d, nn.BatchNorm2d, nn.InstanceNorm2d)
        if isinstance(m, conv_layers):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, norm_layers):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x) -> Union[torch.Tensor, List[torch.Tensor]]:
        # --- Encoder ---
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        
        # --- Bottleneck ---
        x = self.bottleneck(x)
        
        # --- Decoder ---
        outputs = []
        for up, dec, head in zip(self.ups, self.decoders, self.heads):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            outputs.append(head(x))
            
        # 
        return outputs if self.training else outputs[-1]
        

if __name__ == "__main__":


    
    model = UNet(Config())
    
    # Fake 3D volume: [Batch, Channels, D, H, W]
    img = torch.randn(1, 1, 64, 64, 64)
    
    # 1. Training Mode
    model.train()
    targets = model(img)
    print(f"Training mode: produced {len(targets)} output scales.")
    for i, t in enumerate(targets):
        print(f"  Scale {i} shape: {t.shape}")
        
    # 2. Inference Mode
    model.eval()
    with torch.no_grad():
        out = model(img)
    print(f"Inference mode shape: {out.shape} (Highest resolution only)")

    from utils import get_mem_report
    
    mem_report = get_mem_report(model, optimizer_type="adamw")
    print(mem_report)