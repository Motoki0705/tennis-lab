# MAE: Masked Autoencoder for Tennis Domain

This module implements Masked Autoencoder (MAE) pre-training for Vision Transformers,
specifically designed for the tennis domain. The pre-trained encoder can be used for
downstream tasks like ball tracking, player pose estimation, and court detection.

## Architecture Highlights

The ViT encoder incorporates the latest Transformer innovations:

- **2D RoPE (Rotary Position Embedding)**: Spatial positional encoding for patches
- **GQA (Grouped-Query Attention)**: Efficient attention with shared KV heads
- **Register Tokens**: Improved attention patterns following "Vision Transformers Need Registers"
- **SwiGLU MLP**: Modern FFN with gated activation
- **RMSNorm (Pre-Norm)**: Stable training with efficient normalization
- **Optional MoE**: Mixture of Experts for larger capacity
- **Optional MLA**: Multi-head Latent Attention from DeepSeek-V2/V3

Token structure: `[CLS, Register_1, ..., Register_R, Patch_1, ..., Patch_N]`

## Features

- **Variable Resolution Training**: Sample images from a configurable resolution range
- **Efficient Video Loading**: Uses decord or OpenCV for fast frame extraction
- **Configurable Architecture**: All hyperparameters exposed via Hydra configs
- **Lightning Integration**: Full PyTorch Lightning support for training

## Quick Start

### Training

```bash
# Basic training
uv run python -m src.mae.scripts.train

# With custom settings
uv run python -m src.mae.scripts.train \
    model=small \
    data.max_resolution=256 \
    training=fast

# High-resolution training with MoE
uv run python -m src.mae.scripts.train \
    model=large \
    data=high_res \
    training.max_epochs=200
```

### Configuration

Main config: `src/mae/configs/train.yaml`

Available presets:
- **Model**: `base` (ViT-B), `small` (ViT-S), `large` (ViT-L + MoE)
- **Data**: `default`, `high_res`
- **Training**: `default` (400 epochs), `fast` (100 epochs)

### Key Configuration Options

```yaml
# Model
model:
  hidden_dim: 768  # Encoder dimension
  num_layers: 12   # Number of transformer blocks
  num_heads: 12    # Query heads
  num_kv_heads: 4  # KV heads for GQA
  num_register_tokens: 4  # Register tokens
  mask_ratio: 0.75  # Masking ratio
  use_moe: false   # Enable MoE
  use_mla: false   # Enable MLA

# Data
data:
  min_resolution: 160  # Min training resolution
  max_resolution: 320  # Max training resolution
  patch_size: 16

# Training
training:
  learning_rate: 1.5e-4
  max_epochs: 400
  warmup_epochs: 40
```

## Data Preparation

Download tennis videos using the WASB download script:

```bash
# Create urls.yaml with video URLs
# Then download:
uv run python -m src.wasb.scripts.generate_dataset.download_videos \
    urls_path=data/tennis/raw/urls.yaml
```

Videos should be placed in `data/tennis/raw/videos/`.

## Using the Pre-trained Encoder

After training, extract the encoder for downstream tasks:

```python
from src.mae.training import MAELightningModule

# Load checkpoint
module = MAELightningModule.load_from_checkpoint("checkpoints/mae-final.ckpt")
encoder = module.get_encoder()

# Use for downstream task
features = encoder(images)  # (B, hidden_dim) if pooling='cls'
```

## Directory Structure

```
src/mae/
├── __init__.py
├── configs/              # Hydra configurations
│   ├── train.yaml        # Main training config
│   ├── model/            # Model configs
│   │   ├── base.yaml
│   │   ├── small.yaml
│   │   └── large.yaml
│   ├── data/             # Data configs
│   │   ├── default.yaml
│   │   └── high_res.yaml
│   └── training/         # Training configs
│       ├── default.yaml
│       └── fast.yaml
├── data/                 # Data loading
│   ├── __init__.py
│   ├── datamodule.py     # Lightning DataModule
│   └── dataset.py        # Video frame dataset
├── models/               # Model implementations
│   ├── __init__.py
│   └── mae_model.py      # MAE encoder-decoder
├── scripts/              # Training scripts
│   ├── __init__.py
│   └── train.py          # Main training script
└── training/             # Training utilities
    ├── __init__.py
    └── lightning_module.py  # Lightning module

```

## References

- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM](https://arxiv.org/abs/2405.04434)
