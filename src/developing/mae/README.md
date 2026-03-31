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

- **Cached-Batch Training**: No padding; preprocessing (decode/augment/resize/normalize) is done in background
- **Efficient Video Decoding**: Uses decord or OpenCV during cache production
- **Configurable Architecture**: All hyperparameters exposed via Hydra configs
- **Lightning Integration**: Full PyTorch Lightning support for training

## Quick Start

### Training

```bash
# Basic training
python -m src.developing.mae.scripts.train

# With custom settings
python -m src.developing.mae.scripts.train \
    model=small \
    data=cached_batches data.bucket_alpha=2.5 \
    training=fast

# Cached-batch training (no padding; preprocessing in background)
python -m src.developing.mae.scripts.train \
    data=cached_batches
```

### Configuration

Main config: `src/developing/mae/configs/train.yaml`

Available presets:
- **Model**: `base` (ViT-B), `small` (ViT-S), `large` (ViT-L + MoE)
- **Data**: `cached_batches`
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
  mode: cached_batches
  cache_root: data/mae/cache
  samples_per_video: 4
  buckets: [256, 320, 384, 448, 512, 640, 768, 1024]
  bucket_alpha: 2.0  # Higher => lower-res heavier
  preprocess:
    patch_size: 16
    scale_min: 0.8
    scale_max: 1.2

# Training
training:
  learning_rate: 1.5e-4
  max_epochs: 400
  warmup_epochs: 40

# Run
run:
  output_dir: outputs/mae
  seed: 42
  gpus: 1
  fast_dev_run: false
  dry_run: false
```

## Data Preparation

Download tennis videos using the WASB download script:

```bash
# Create urls.yaml with video URLs
# Then download:
python -m src.tasks.wasb.scripts.generate_dataset.download_videos \
    urls_path=data/tennis/raw/urls.yaml
```

Videos should be placed in `data/tennis/raw/videos/`.

## Using the Pre-trained Encoder

After training, extract the encoder for downstream tasks:

```python
from src.developing.mae.training import MAELightningModule

# Load checkpoint
module = MAELightningModule.load_from_checkpoint("checkpoints/mae-final.ckpt")
encoder = module.get_encoder()

# Use for downstream task
features = encoder(images)  # (B, hidden_dim) if pooling='cls'
```

## Directory Structure

```
src/developing/mae/
├── __init__.py
├── configs/              # Hydra configurations
│   ├── train.yaml        # Main training config
│   ├── model/            # Model configs
│   │   ├── base.yaml
│   │   ├── small.yaml
│   │   └── large.yaml
│   ├── data/             # Data configs
│   │   └── cached_batches.yaml
│   └── training/         # Training configs
│       ├── default.yaml
│       └── fast.yaml
├── data/                 # Data loading
│   ├── __init__.py
│   ├── datamodule.py     # Lightning DataModule
│   ├── dataset_cached.py # Cached-batch dataset
│   ├── planning.py       # Epoch/bucket planning
│   └── producer.py       # Cache producer
├── models/               # Model implementations
│   ├── __init__.py
│   └── mae_model.py      # MAE encoder-decoder
├── scripts/              # Training scripts
│   ├── __init__.py
│   ├── train.py          # Main training script
│   └── produce_epoch_cache.py # Cache generation helper
└── training/             # Training utilities
    ├── __init__.py
    ├── lightning_module.py  # Lightning module
    └── epoch_cache_callback.py # Background cache callback

```

## References

- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM](https://arxiv.org/abs/2405.04434)
