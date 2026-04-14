# [ON-DEVICE MEDICAL SEGMENTATION] - Workspace Instructions

## Project Overview

The goal of this project is to  build a repository to: 

1 - Provide a clean, minimal codebase to train several 2D/3D UNet variants for medical segmentation
2 - Allow to explore and evaluate the performance trade-offs of different quantizations and memory optimizations techniques
3 - Export models to edge-friendly formats (ONNX) for deployment on mobile embedded devices
4 - Allow to produce a "segmentation summary", a radiology like report that summarizes the findings of the segmentation results in a human readable format.

## Current State & Priorities

[TBD - update this with current priorities]

## Code Organization

Follow this structure when adding or modifying code. Do not mix responsibilities across files.

```
src/
├── config.py         # Minimal runtime configuration
├── model.py          # UNet definitions
├── dataset.py        # Dataset loading
├── transforms.py     # Preprocessing and augmentation
├── loss.py           # Loss functions
├── optim.py          # Optimizer and scheduler setup
├── train.py          # Training loop
├── eval.py           # Evaluation and metrics
├── export.py         # ONNX export
├── quantize.py       # Quantization utilities and experiments
├── report.py         # Segmentation summary generation
└── utils.py          # Small shared utilities

config/               # Minimal experiment configs
scripts/              # Data preparation and helper scripts
tests/ 
data/                 # Raw and processed data  
```


### User interaction

- Remove filler words (e.g., just, really, basically, actually, simply)
- Omit pleasantries (e.g., sure, certainly, of course)
- Prefer concise wording and short synonyms
- Eliminate hedging (e.g., avoid might, could, worth considering)
- Use direct, assertive statements
- Prioritize clarity over tone

### Feedback & Critique

- Be direct; do not soften criticism
- Prioritize highest-impact issues first
- Distinguish between must-fix vs nice-to-have
- Suggest exact improvements, not general advice
- Avoid rewriting everything unless necessary—target key changes

### Coding

- Provide minimal, readable solutions
- Use clear, descriptive, unambiguous names
- Prefer straightforward implementations over abstraction layers
- Avoid unnecessary dependencies; rely on essential libraries only
- Eliminate hidden complexity and implicit behavior
- Keep functions small and single-purpose
- Validate inputs and fail early with explicit errors
- Use docstrings for purpose and intent; avoid commenting obvious code


### Testing

- Prefer simple, local validation in `if __name__ == "__main__":` blocks.
- Test on a small, representative subset of inputs (2–5 cases), not the full dataset.
- Never iterate over all files or large directories for validation.
- Focus on edge cases and typical cases, not exhaustive coverage.
- Do not generate generic unit tests without a clear, specific purpose.
- Keep tests minimal, fast, and deterministic.
- Fail fast with clear assertions; avoid silent checks.

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

