"""Loader for the DINO FPN v2 court keypoint estimator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml

from .dino_fpn_utils import align_and_load, load_lightning_state_dict, strip_prefix
from .model import create_model


def _to_tuple(data: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return tuple(int(v) for v in data)
    return (int(data),)


@dataclass
class DinoFpnV2LoadConfig:
    checkpoint_path: str
    repo_dir: str = "third_party/dinov3"
    entry: str = "dinov3_vits16"
    weights: str | None = None
    freeze: bool = True
    vit_layers: Sequence[int] = (11,)
    fpn_channels: int = 256
    num_keypoints: int = 15
    decoder_base_channels: int = 256
    decoder_channels: Sequence[int] = (256, 128, 64)
    decoder_upsample_mode: str = "bilinear"
    decoder_final_activation: str | None = None
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    pad_to_multiple: int | None = 16
    resize_long_side: int | None = None
    device: str = "cuda"
    strict: bool = False
    remove_prefix: str = "model."
    allow_partial: bool = True

    def __post_init__(self) -> None:
        self.vit_layers = _to_tuple(self.vit_layers)
        self.decoder_channels = _to_tuple(self.decoder_channels)
        self.mean = tuple(float(x) for x in self.mean)
        self.std = tuple(float(x) for x in self.std)
        if isinstance(self.weights, str) and not self.weights.strip():
            self.weights = None

    @classmethod
    def from_yaml(cls, path: str) -> DinoFpnV2LoadConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

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
    def __init__(self, multiple: int) -> None:
        self.multiple = int(multiple)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, height, width = tensor.shape
        pad_h = (self.multiple - (height % self.multiple)) % self.multiple
        pad_w = (self.multiple - (width % self.multiple)) % self.multiple
        if pad_h == 0 and pad_w == 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)


def _build_transform(cfg: DinoFpnV2LoadConfig) -> Callable:
    ops: list[Callable] = []
    if cfg.resize_long_side is not None:
        ops.append(T.Resize(cfg.resize_long_side, max_size=None))
    ops.extend([T.ToTensor(), T.Normalize(mean=cfg.mean, std=cfg.std)])
    transform = T.Compose(ops)

    if cfg.pad_to_multiple is None:
        return transform

    padder = _PadToMultiple(cfg.pad_to_multiple)

    def _wrapped(image):
        tensor = transform(image)
        return padder(tensor)

    return _wrapped


def load_dino_fpn_v2_with_ckpt(
    cfg: DinoFpnV2LoadConfig,
) -> tuple[torch.nn.Module, Callable, torch.device]:
    """Instantiate the DINO FPN v2 model, load weights, and return (model, transform, device)."""

    checkpoint = Path(cfg.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    repo_dir = Path(cfg.repo_dir).expanduser().resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repo directory not found: {repo_dir}")

    weights_path = None
    if cfg.weights is not None:
        weights_candidate = Path(cfg.weights).expanduser().resolve()
        if not weights_candidate.is_file():
            raise FileNotFoundError(f"Backbone weights file not found: {weights_candidate}")
        weights_path = str(weights_candidate)

    device = _select_device(cfg.device)

    model = create_model(
        backbone={
            "repo_dir": str(repo_dir),
            "entry": cfg.entry,
            "weights": weights_path,
            "freeze": cfg.freeze,
            "vit_layers": list(cfg.vit_layers),
            "fpn_channels": cfg.fpn_channels,
        },
        decoder={
            "num_keypoints": cfg.num_keypoints,
            "base_channels": cfg.decoder_base_channels,
            "inner_channels": list(cfg.decoder_channels),
            "upsample_mode": cfg.decoder_upsample_mode,
            "final_activation": cfg.decoder_final_activation,
        },
    ).to(device)

    transform = _build_transform(cfg)

    raw_state = load_lightning_state_dict(str(checkpoint))
    if cfg.remove_prefix:
        raw_state = strip_prefix(raw_state, prefix=cfg.remove_prefix)

    if cfg.allow_partial:
        align_and_load(model, raw_state, strict=cfg.strict)
    else:
        model.load_state_dict(raw_state, strict=cfg.strict)

    model.eval()
    return model, transform, device


__all__ = [
    "DinoFpnV2LoadConfig",
    "load_dino_fpn_v2_with_ckpt",
]
