"""Person appearance encoders behind one interface.

Every encoder maps RGB crops, already resized to its ``input_size`` and scaled
to ``[0, 1]``, to L2-normalized embeddings. Normalization constants belong to
the encoder. Weights are loaded with an explicit key check: only the identity
classifier of a Re-ID checkpoint may be absent from the model, nothing else.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


class AppearanceEncoder(Protocol):
    name: str
    input_size: tuple[int, int]  # height, width

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` RGB in ``[0, 1]`` -> ``(N, D)`` unit-norm embeddings."""
        ...


def _normalized(crops: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    if crops.ndim != 4 or crops.shape[1] != 3:
        raise ValueError("Crops must be (N, 3, H, W)")
    device = crops.device
    return (crops - torch.tensor(mean, device=device)[:, None, None]) / torch.tensor(std, device=device)[:, None, None]


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"{path} does not hold a state dict")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _load_checked(module: nn.Module, state: Mapping[str, torch.Tensor], *, optional_missing: tuple[str, ...]) -> None:
    result = module.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith(optional_missing)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Encoder weights do not match: missing={missing}, unexpected={result.unexpected_keys}")


@dataclass
class OSNetEncoder:
    """torchreid OSNet / OSNet-AIN trained for person Re-ID (512-d)."""

    name: str
    architecture: str  # torchreid model name, e.g. "osnet_ain_x1_0"
    checkpoint: Path
    device: str = "cpu"
    input_size: tuple[int, int] = (256, 128)

    def __post_init__(self) -> None:
        from torchreid.reid.models import build_model

        self.model = build_model(name=self.architecture, num_classes=1, pretrained=False)
        state = {key: value for key, value in _load_state(self.checkpoint).items() if not key.startswith("classifier.")}
        _load_checked(self.model, state, optional_missing=("classifier.",))
        self.model.eval().to(self.device)

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(_normalized(crops.to(self.device), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        return F.normalize(features.float(), dim=-1).cpu()


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(width, width * 4)), ("gelu", _QuickGELU()),
                                              ("c_proj", nn.Linear(width * 4, width))]))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln_1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(*[_ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.resblocks(x)
        return output


class _ClipVisual(nn.Module):
    """The CLIP ViT-B/16 visual tower as fine-tuned by CLIP-ReID (non-overlapping 16 px patches)."""

    def __init__(self, grid: tuple[int, int], width: int = 768, layers: int = 12, heads: int = 12, output_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=16, stride=16, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(grid[0] * grid[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, output_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.class_embedding.expand(len(x), 1, -1), x), 1) + self.positional_embedding
        x = self.transformer(self.ln_pre(x).transpose(0, 1)).transpose(0, 1)
        cls = self.ln_post(x[:, 0])
        return cls, cls @ self.proj


@dataclass
class ClipReIDEncoder:
    """CLIP-ReID (Li et al., 2023) image encoder; the test feature concatenates the CLS token and its projection (1280-d)."""

    name: str
    checkpoint: Path
    device: str = "cpu"
    input_size: tuple[int, int] = (256, 128)

    def __post_init__(self) -> None:
        self.model = _ClipVisual((self.input_size[0] // 16, self.input_size[1] // 16))
        prefix = "image_encoder."
        state = {key.removeprefix(prefix): value for key, value in _load_state(self.checkpoint).items() if key.startswith(prefix)}
        _load_checked(self.model, state, optional_missing=())
        self.model.eval().to(self.device)

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cls, projected = self.model(_normalized(crops.to(self.device), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return F.normalize(torch.cat((cls, projected), -1).float(), dim=-1).cpu()


@dataclass
class DINOv3Encoder:
    """Generic self-supervised DINOv3 features (not trained for Re-ID): normalized CLS token."""

    name: str
    backbone_name: str
    repository: Path
    checkpoint: Path
    device: str = "cpu"
    input_size: tuple[int, int] = (256, 128)

    def __post_init__(self) -> None:
        from src.utils.models.loading.dinov3 import load_dinov3_backbone

        self.backbone = load_dinov3_backbone(repository_path=self.repository, checkpoint_path=self.checkpoint,
                                             backbone_name=self.backbone_name, strict=True)
        self.backbone.module.eval().to(self.device)

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.module.forward_features(_normalized(crops.to(self.device), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        return F.normalize(features["x_norm_clstoken"].float(), dim=-1).cpu()


ENCODER_CANDIDATES = ("osnet_ain_x1_0_msmt17", "osnet_x1_0_msmt17", "clipreid_vitb16_market1501", "dinov3_vits16", "dinov3_vitb16")
"""Encoders compared for #933. Re-ID weights live under ``<checkpoint_root>/player_association``; DINOv3 under ``<external_root>/dinov3``."""


def build_encoder(name: str, *, checkpoint_root: Path, external_root: Path, device: str) -> AppearanceEncoder:
    reid = checkpoint_root / "player_association"
    dinov3 = external_root / "dinov3"
    if name == "osnet_ain_x1_0_msmt17":
        return OSNetEncoder(name, "osnet_ain_x1_0", reid / "osnet_ain_x1_0_msmt17.pth", device)
    if name == "osnet_x1_0_msmt17":
        return OSNetEncoder(name, "osnet_x1_0", reid / "osnet_x1_0_msmt17_combineall.pth", device)
    if name == "clipreid_vitb16_market1501":
        return ClipReIDEncoder(name, reid / "person_vit_clip_reid.pth", device)
    if name == "dinov3_vits16":
        return DINOv3Encoder(name, name, dinov3, dinov3 / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", device)
    if name == "dinov3_vitb16":
        return DINOv3Encoder(name, name, dinov3, dinov3 / "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", device)
    raise ValueError(f"Unknown appearance encoder {name!r}; candidates: {ENCODER_CANDIDATES}")
