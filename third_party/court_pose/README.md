# Court Pose Inference

The `dino_fpn` subdirectory contains the COAT (DINO + FPN) model used for
predicting 2D tennis court keypoints. Use
`dino_fpn/dino_fpn_loader.py` to rebuild the pure model, load the Lightning
checkpoint, and obtain the preprocessing pipeline.

## Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA; CUDA is recommended for speed but not required)
- `torchvision` and `Pillow` for image transforms
- A trained Lightning checkpoint compatible with the provided config

## Running inference

```python
from pathlib import Path

import torch
from PIL import Image

from trained_models.court_pose.dino_fpn.dino_fpn_loader import (
    CoatLoadConfig,
    load_coat_with_ckpt,
)

cfg = CoatLoadConfig.from_yaml(
    Path("trained_models/court_pose/dino_fpn/configs/coat_config.yaml")
)
cfg.checkpoint_path = "/path/to/lightning.ckpt"

model, transform, device = load_coat_with_ckpt(cfg)

image = Image.open("frame.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)

model.eval()
with torch.inference_mode():
    heatmaps = model(tensor)

# Convert heatmaps to (x, y) coordinates – simple argmax baseline
coords = []
for channel in heatmaps[0]:
    value, index = torch.max(channel.reshape(-1), dim=0)
    y = (index // channel.shape[-1]).item()
    x = (index % channel.shape[-1]).item()
    coords.append((x, y, float(value)))

print("Court keypoints:", coords)
```

### Tips

- The loader normalises pixels with ImageNet statistics and optionally pads to
  a multiple of 16 if configured. Make sure you do not apply extra
  preprocessing.
- For higher accuracy replace the argmax post-processing with soft-argmax or
  Gaussian peak fitting.
- If you need batched inference simply stack multiple transformed tensors into
  a single batch before calling `model`.

## DINO FPN v2 variant

The `dino_fpn_v2` folder packages the upgraded DINOv3 + FPN court pose model. Load it with
`dino_fpn_v2/dino_fpn_v2_loader.py` and the matching dataclass `DinoFpnV2LoadConfig`:

```python
from trained_models.court_pose.dino_fpn_v2.dino_fpn_v2_loader import (
    DinoFpnV2LoadConfig,
    load_dino_fpn_v2_with_ckpt,
)

cfg = DinoFpnV2LoadConfig.from_yaml("trained_models/court_pose/dino_fpn_v2/config.yaml")
cfg.checkpoint_path = "/path/to/dino_fpn_v2.ckpt"

model, transform, device = load_dino_fpn_v2_with_ckpt(cfg)
```

This variant initialises a DINOv3 backbone through `torch.hub.load`, so ensure `third_party/dinov3/` is present (or
override `cfg.repo_dir`). The transform pipeline mirrors the v1 loader; its output heatmaps align with the 1:1 image
resolution after interpolation.
