#!/bin/bash
# Medical Segmentation Training Pipeline
# Nanochat-inspired: single script to run the complete pipeline

set -e  # Exit on error

# Configuration 
DATA_DIR="./data/processed"
CONFIG=${1:-"config/2d_axi.yaml"}  # Use first arg or default
CHECKPOINT_DIR="./checkpoints"

echo "🔬 Medical Segmentation Training Pipeline"
echo "==========================================="
echo "Config: $CONFIG"
echo "Data dir: $DATA_DIR"
echo ""

# Step 1: Check if data exists, if not suggest preparation
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "❌ No processed data found in $DATA_DIR"
    echo "💡 To prepare data, run:"
    echo "   python scripts/prepare.py --raw_dir ./BraTS2021 --save_dir $DATA_DIR"
    echo ""
    exit 1
fi

# Step 2: Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Step 3: Train model (nanochat style: simple command)
echo "🚀 Starting training with $CONFIG"
echo "   Output: $CHECKPOINT_DIR"
echo ""

python src/train.py --config="$CONFIG"

# Step 4: Success message
echo ""
echo "✅ Training completed!"
echo "📁 Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "💡 Next steps:"
echo "   - Evaluate: python src/eval.py --checkpoint $CHECKPOINT_DIR/best_model.pth"
echo "   - Different config: bash train.sh config/3d_vol.yaml" 
echo "   - Mobile export: python scripts/export_mobile.py --checkpoint $CHECKPOINT_DIR/best_model.pth"