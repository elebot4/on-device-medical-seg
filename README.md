# 🔬 Medical Segmentation Mobile

TBD

## ✨ Key Features (Karpathy-Inspired)

- 🎯 **Explicit Control**: Specify exactly what you want - no hidden auto-calculations
- 📦 **Minimal & Hackable**: No configuration monsters or factory patterns
- 🚀 **Working Baseline**: Runs end-to-end without extensive setup
- 📱 **Mobile-First**: Optimized for resource-constrained deployment

## 🚀 Quick Start  

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train with single complexity dial (nanochat-style!)
python src/train.py --config config/2d_axi.yaml

# 3. Or use the complete pipeline
bash train.sh config/mobile.yaml
```

## 🏛️ Karpathy-Style Configuration

Clean, explicit parameters - no magic auto-calculations:

```yaml
# config/my_config.yaml
num_stages: 4      # Number of encoder/decoder stages
base_chs: 32       # Base channel count
lr: 3e-4           # Learning rate
batch_size: 8      # Batch size
deep_supervision: true

# Specify exactly what you want!
```

### Model Configurations

| Config | Use Case | Stages | Channels | Batch |
|--------|----------|--------|----------|-------|
| `2d_axi.yaml` | **Default 2D** | 4 | 32 | 8 |

## 📁 Project Structure (Clean & Minimal)

```
src/
├── config.py         # Single-dial configuration (nanochat-style)
├── model.py          # 3D/2D UNet with deep supervision  
├── dataset.py        # Data loading + augmentations
├── loss.py           # Dice + composite losses
├── optim.py          # Smart optimizer factory  
├── train.py          # Training loop with AMP
└── utils.py          # Memory utilities

config/               # Ready-to-use configurations
├── 2d_axi.yaml       # Complexity 3 - 2D axial (default)
└──
```

## 🎯 Usage Examples

### Training Different Scales
```bash  
# Quick mobile model (1 min training)
python src/train.py --config config/mobile.yaml

# Standard 2D model (good accuracy/speed balance)
python src/train.py --config config/2d_axi.yaml  

# High-res research model  
python src/train.py --config config/xl.yaml
```

```bash
# Fine-tune specific parameters
python src/train.py --config config/2d_axi.yaml lr=0.001 batch_size=4

# Experiment with architecture
python src/train.py --config config/2d_axi.yaml num_stages=5 base_chs=64
```

## 🔧 Data Preparation

```bash
# Prepare BraTS dataset (or similar medical data)
python scripts/prepare.py --raw_dir ./BraTS2021 --save_dir ./data/processed
```

Expected structure:
```
data/processed/
├── images/           # .npy files
│   ├── case001.npy
│   └── case002.npy  
└── masks/            # .npy files
    ├── case001.npy
    └── case002.npy
```

## 📜 License

MIT License - fork freely!

## 🙏 Acknowledgements  

- [karpathy/nanochat](https://github.com/karpathy/nanochat) - Design philosophy inspiration
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) - Medical segmentation insights
- Medical imaging community datasets
