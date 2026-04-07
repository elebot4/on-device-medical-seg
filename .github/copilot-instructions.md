# Medical Segmentation Mobile - Workspace Instructions

## Project Overview

TBD

## Architecture Philosophy (karpathy-inspired)

- **Minimal**: No configuration monsters and minimal factory patterns. The code should be straightforward to read and modify.
- **Explicit Control**: Direct parameter control - specify exactly what you want
- **Explicit Precision**: Direct precision management, no hidden autocast
- **Working Baseline**: Code runs end-to-end without extensive configuration

## Current State & Priorities

TBD

## Code Organization

```
src/
├── config.py         # Central dataclass configuration (nanochat-style)
├── model.py          # 3D/2D UNet with deep supervision
├── dataset.py        # Data loading with hardware alignment
├── transforms.py     # Medical image augmentations
├── loss.py           # Dice + Composite losses
├── optim.py          # Smart optimizer factory
├── train.py          # Training loop with AMP
└── utils.py          # Memory reporting utilities

config/               # YAML configurations for different views
scripts/              # Data preparation utilities
```

## Coding Standards

### Configuration Pattern
- Use dataclass-based config (see `src/config.py`)
- Single source of truth with YAML + CLI override capability
- Explicit parameter specification - no auto-calculation

### Model Architecture
- Support both 2D/3D operations
- Support deep supervision with strided masking for pyramid levels

### Training Loop
- Mixed precision training (AMP) by default

### Optimizer settings
- Smart weight decay (decouple conv weights from biases/norms)

### File Naming & Imports
- Use absolute imports: `from src.module import function`
- Configuration files: `config/{view_type}.yaml` (e.g., `2d_axi.yaml`)
- Scripts focus on single tasks: `train.py`, `prepare.py`, `eval.py`

## Development Workflow

TBD

### Training Pipeline
```bash
# 1. Prepare data
python scripts/prepare.py --raw_dir ./BraTS2021 --save_dir ./data/processed

# 2. Train model (single complexity dial)
python src/train.py --config config/2d_axi.yaml
# or with overrides
python src/train.py --config config/2d_axi.yaml --lr 0.001 --batch_size 4

# 3. Evaluate
python src/eval.py --checkpoint checkpoints/best_model.pth --test_dir ./data/test
```

### Expected Quick Commands
- `bash train.sh` - Complete training pipeline
- `python src/train.py --help` - See all configuration options
- `docker build -t med_seg .` - Build training container

## Key Patterns to Follow

### 1. Karpathy-Style Configuration
```python
@dataclass
class Config:
    # Explicit parameters - no magic auto-calculation
    num_stages: int = 4
    base_chs: int = 32
    lr: float = 3e-4
    
    # Clean validation in __post_init__
    def __post_init__(self):
        if self.milestones is None:
            self.milestones = [0.5, 0.75]
```

### 2. Hardware-Aware Design
```python
# Align dimensions for efficient GPU computation
def pad_to_multiple(size, multiple=16):
    return ((size + multiple - 1) // multiple) * multiple
```

### 4. Medical Domain Specifics
- **Volume Assumptions**: `.npy` files in `{data_dir}/images/` and `{data_dir}/masks/`
- **Deep Supervision**: Multi-scale loss with strided targets
- **Augmentation Chain**: Spatial → Intensity → Resolution transforms

## Anti-Patterns to Avoid

❌ **Auto-calculated Dependencies**: Hidden parameter calculations  
✅ **Explicit Parameters**: Specify exactly what you want

❌ **Single Dial Abstraction**: Magic auto-scaling based on one parameter  
✅ **Direct Control**: Set `num_stages: 4`, `base_chs: 32` explicitly

❌ **Factory Patterns**: `ModelFactory.create(type="unet", variant="3d")`  
✅ **Direct Instantiation**: `UNet3D(config)` with runtime operator selection

❌ **Scattered Configs**: Parameters spread across multiple files  
✅ **Central Config**: Single dataclass with YAML/CLI override

## Testing Strategy

TBD

## Deployment Pipeline

1. **Training**: Standard PyTorch model  
2. **Quantization**: Post-training quantization for mobile  
3. **Export**: ONNX, TensorFlow Lite, CoreML formats  
4. **Validation**: Accuracy retention checks after optimization

## Quick Reference

### Common Tasks
- **Add new model architecture**: Extend `model.py` with operator selection pattern
- **New augmentation**: Add to `transforms.py` augmentation chain
- **Custom loss**: Implement in `loss.py` with deep supervision support
- **Different dataset**: Modify `dataset.py` loading logic, keep same interface

### Configuration Examples
```yaml
# config/mobile_2d.yaml - Optimized for mobile deployment
model_depth: 3        # Smaller model
base_channels: 16     # Reduced capacity
mixed_precision: true # Memory efficiency
deep_supervision: false # Simplify for mobile
```

*Following nanochat principles: This should be a "cohesive, minimal, readable, hackable, maximally-forkable strong baseline" for medical image segmentation.*