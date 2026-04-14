import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Building Blocks

class Block(nn.Module):
    """
    Residual block with configurable Conv-Norm-Activation layers.
    Supports both 2D and 3D operations based on input shape configuration.
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels  
        spatial_dims: Number of spatial dimensions (2 or 3)
        norm_type: Type of normalization ('group', 'batch', 'instance', 'none')
        act_type: Type of activation ('relu', 'gelu', 'leaky')
        dropout: Dropout rate
        norm_groups: Number of groups for GroupNorm
    """
    def __init__(self, in_ch, out_ch, spatial_dims, norm_type='group', act_type='relu', dropout=0.1, norm_groups=8):
        super().__init__()
        
        # 1. Local aliasing and configuration of operators based on spatial dimensions
        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        dropout_layer = nn.Dropout3d if spatial_dims == 3 else nn.Dropout2d
        
        if norm_type == 'group':
            norm = lambda c: nn.GroupNorm(norm_groups, c)
        elif norm_type == 'batch':
            norm = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        elif norm_type == 'instance':
            norm = nn.InstanceNorm3d if spatial_dims == 3 else nn.InstanceNorm2d
        else:
            norm = nn.Identity

        if act_type == 'relu':
            act = nn.ReLU
        elif act_type == 'gelu':
            act = nn.GELU
        else:
            act = lambda: nn.LeakyReLU(0.01)

        # 2. Architectural definition
        self.shortcut = conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        self.net = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch),
            act(),
            dropout_layer(dropout),
            conv(out_ch, out_ch, 3, padding=1, bias=False),
            norm(out_ch),
            act(),
        )


    def forward(self, x):
        return self.net(x) + self.shortcut(x)

# -----------------------------------------------------------------------------
# Main Model

class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation.
    
    Supports both 2D and 3D operations with configurable:
    - Number of encoder/decoder stages
    - Base number of channels with 2x scaling per stage
    - Deep supervision with multi-scale outputs
    - Residual blocks with skip connections
    """
    def __init__(self, input_shape, in_channels, out_channels, num_stages, base_chs, 
                 norm_type='group', act_type='relu', dropout=0.1, norm_groups=8, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Local aliasing and configuration of operators based on spatial dimensions
        conv = nn.Conv3d if len(input_shape) == 3 else nn.Conv2d
        conv_t = nn.ConvTranspose3d if len(input_shape) == 3 else nn.ConvTranspose2d
        spatial_dims = len(input_shape)
        
        # Channel schedule: e.g., [32, 64, 128, 256]
        chs = [base_chs * (2**i) for i in range(num_stages)]
        
        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        curr_in = in_channels
        for i in range(num_stages - 1):
            self.encoders.append(
                Block(curr_in, chs[i], spatial_dims, norm_type, act_type, dropout, norm_groups)
            )
            self.downs.append(conv(chs[i], chs[i], kernel_size=2, stride=2))
            curr_in = chs[i]

        # --- Bottleneck ---
        self.bottleneck = Block(chs[-2], chs[-1], spatial_dims, norm_type, act_type, dropout, norm_groups)
        
        # --- Decoder & Deep Supervision Heads ---
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList() # One head for each decoder stage for deepsupervision

        # We build the decoder from deepest to shallowest
        for i in reversed(range(num_stages - 1)):
            self.ups.append(conv_t(chs[i+1], chs[i], kernel_size=2, stride=2))
            self.decoders.append(
                Block(chs[i] * 2, chs[i], spatial_dims, norm_type, act_type, dropout, norm_groups)
            )
            self.heads.append(conv(chs[i], out_channels, kernel_size=1))

        self.apply(self._init_weights)
        print(f"UNet initialized: {sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def _init_weights(self, m):
        conv_layers = (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)
        norm_layers = (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d, nn.BatchNorm2d, nn.InstanceNorm2d)
        if isinstance(m, conv_layers):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, norm_layers):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
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
            
        # Return all outputs for deep supervision during training, only final for inference
        return outputs if (self.training and self.deep_supervision) else outputs[-1]
        

if __name__ == "__main__":
    # Simple test of the model
    # Set some test parameters
    input_shape = (64, 64, 64)
    in_channels = 1
    out_channels = 2  
    num_stages = 4
    base_chs = 32
    norm_type = 'group'
    act_type = 'relu'
    dropout = 0.1
    norm_groups = 8
    deep_supervision = True
    
    model = UNet()
    x = torch.randn(1, in_channels, *input_shape)
    
    print(f"Input shape: {x.shape}")
    
    model.train()
    outputs = model(x)
    if isinstance(outputs, list):
        print(f"Training mode - Deep supervision outputs: {[o.shape for o in outputs]}")
    else:
        print(f"Training mode output: {outputs.shape}")
    
    model.eval() 
    output = model(x)
    print(f"Inference mode output: {output.shape}")


    
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