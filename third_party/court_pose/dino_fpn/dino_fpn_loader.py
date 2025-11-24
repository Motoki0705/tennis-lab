# coat_loader.py
"""
Loader that:
  1) Instantiates the pure DinoVitHeatmap model (no Lightning wrapper)
  2) Overwrites its weights using a Lightning checkpoint (after removing 'model.')
  3) Returns: model, a torchvision transform pipeline, and the device

Quickstart
----------
from coat_loader import CoatLoadConfig, load_coat_with_ckpt

cfg = CoatLoadConfig.from_yaml("coat_config.yaml")
model, transform, device = load_coat_with_ckpt(cfg)

# Inference example (single image)
# --------------------------------
from PIL import Image
import torch

img = Image.open("example.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

model.eval()
with torch.inference_mode():
    heatmaps = model(tensor)  # [1, K, H, W] at input resolution

# Get argmax keypoints per channel (simple baseline)
# (For better accuracy, consider soft-argmax or Gaussian peak fitting.)
k = heatmaps.shape[1]
coords, scores = [], []
for c in range(k):
    hm = heatmaps[0, c]
    v, idx = torch.max(hm.view(-1), dim=0)
    y = idx // hm.shape[-1]
    x = idx % hm.shape[-1]
    coords.append((int(x.item()), int(y.item())))
    scores.append(float(v.item()))
print({"coords_xy": coords, "scores": scores})

Notes
-----
- Input size can be arbitrary; the decoder upsamples to the original size.
- For ViT-S/16 backbones, multiples of 16 are generally ideal, but the module
  handles non-16 multiples via final interpolate.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torchvision.transforms as T
import yaml

from .dino_fpn_utils import align_and_load, load_lightning_state_dict, strip_prefix

# Import your pure model (adjust the path to where DinoVitHeatmap lives)
# e.g., from mypkg.models.coat.model.model import DinoVitHeatmap
from .model.model import DinoVitHeatmap  # <-- adjust if needed


@dataclass
class CoatLoadConfig:
    # --- Paths ---
    checkpoint_path: str

    # --- Model params (match your training) ---
    heatmap_channels: int = 15
    decoder_channels: list | None = None
    backbone_name: str = "dinov3_vits16"
    weights_path: str | None = "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    # --- Transforms (DINO/Imagenet-style defaults) ---
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    pad_to_multiple: int | None = 16  # set None to disable padding

    # --- Runtime ---
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    strict: bool = False
    remove_prefix: str = "model."
    allow_partial: bool = True

    # Optional resize (keep None to use raw image size)
    resize_long_side: int | None = None  # e.g., 1024

    @classmethod
    def from_yaml(cls, path: str) -> CoatLoadConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)


def _select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _PadToMultiple:
    """Pad to make H and W multiples of `m` (right/bottom padding)."""

    def __init__(self, m: int):
        self.m = int(m)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [C, H, W]
        c, h, w = img.shape
        pad_h = (self.m - (h % self.m)) % self.m
        pad_w = (self.m - (w % self.m)) % self.m
        if pad_h == 0 and pad_w == 0:
            return img
        return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def _build_transform(cfg: CoatLoadConfig) -> Callable:
    """Compose torchvision transforms -> tensor in [C,H,W] normalized."""
    ops = []
    if cfg.resize_long_side is not None:
        ops.append(T.Resize(cfg.resize_long_side, max_size=None))  # keep aspect ratio (torch>=0.18)
    ops += [T.ToTensor(), T.Normalize(mean=cfg.mean, std=cfg.std)]
    transform = T.Compose(ops)

    if cfg.pad_to_multiple is None:
        return transform

    padder = _PadToMultiple(cfg.pad_to_multiple)

    def _wrapped(img):
        t = transform(img)  # [C,H,W]
        return padder(t)

    return _wrapped


def load_coat_with_ckpt(
    cfg: CoatLoadConfig,
) -> tuple[DinoVitHeatmap, Callable, torch.device]:
    """
    Build a pure DinoVitHeatmap model and overwrite its weights from a Lightning checkpoint.

    Returns
    -------
    model : DinoVitHeatmap
        The model with checkpoint weights loaded (intersecting keys by default).
    transform : Callable
        A torchvision preprocessing callable: PIL.Image|ndarray -> normalized torch.FloatTensor[C,H,W].
    device : torch.device
        The device the model has been moved to.
    """
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    device = _select_device(cfg.device)

    # 1) Instantiate pure model (no Lightning wrapper)
    model = DinoVitHeatmap(
        num_keypoints=cfg.heatmap_channels,
        decoder_channels=cfg.decoder_channels,
        backbone_name=cfg.backbone_name,
        weights_path=cfg.weights_path,
    ).to(device)

    # 2) Build transform
    transform = _build_transform(cfg)

    # 3) Load Lightning checkpoint -> strip prefix -> align & load
    raw_sd = load_lightning_state_dict(cfg.checkpoint_path)
    if cfg.remove_prefix:
        raw_sd = strip_prefix(raw_sd, prefix=cfg.remove_prefix)

    if cfg.allow_partial:
        align_and_load(model, raw_sd, strict=cfg.strict)
    else:
        model.load_state_dict(raw_sd, strict=cfg.strict)

    model.eval()
    return model, transform, device
