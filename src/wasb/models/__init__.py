"""Models for WASB tennis dataset generation."""

from __future__ import annotations

from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .clip_segmenter import ClipSegmenter, RuleBasedClipSegmenter
from .hrnet import HRNet
from .hrcnet import HRCNet
from .rnn_hrnet import HRNetConvGRU
from .trajectory_completer import (
    BiLSTMCompleter,
    CompletionResult,
    HybridCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    create_completer,
)


def _hrnet_handlers() -> tuple[Callable, Callable]:
    def prepare_frames(frames):
        if getattr(frames, "dim", None) is not None and frames.dim() == 5:
            b, t, c, h, w = frames.shape
            return frames.view(b, t * c, h, w)
        return frames

    def extract_heatmaps(outputs):
        if isinstance(outputs, Tensor):
            return outputs  
        raise TypeError("Unsupported model output type for HRNet.")

    return prepare_frames, extract_heatmaps


def _hrnet_gru_handlers() -> tuple[Callable, Callable]:
    def prepare_frames(frames):
        if getattr(frames, "dim", None) is not None and frames.dim() != 5:
            raise ValueError(
                f"HRNetConvGRU expects input shape [B, T, C, H, W], got {getattr(frames, 'shape', None)}"
            )
        return frames

    def extract_heatmaps(outputs):
        if isinstance(outputs, tuple):
            pred, _ = outputs
            if not isinstance(pred, Tensor):
                raise TypeError("First element of HRNetConvGRU output must be a Tensor.")
            return pred
        raise TypeError("Unsupported model output type for HRNetConvGRU.")

    return prepare_frames, extract_heatmaps

__factory: dict[str, Any] = {
    "hrnet": lambda cfg: (HRNet(cfg), _hrnet_handlers()),
    "hrcnet": lambda cfg: (
        HRCNet(
            in_channels=3 * cfg.get("frames_in", 1),
            out_channels=cfg.get("frames_out", 1),
            high_channels=cfg.get("high_channels", 64),
            low_channels=cfg.get("low_channels", 64),
            num_stages=cfg.get("num_stages", 3),
            high_block=cfg.get("high_block", "BASIC"),
            low_block=cfg.get("low_block", "BASIC"),
            num_high_blocks=cfg.get("num_high_blocks", 2),
            num_low_blocks=cfg.get("num_low_blocks", 1),
            upsample_mode=cfg.get("upsample_mode", "nearest"),
            downsample_kwargs=cfg.get("downsample_kwargs", {}),
            transformer_kwargs=cfg.get("transformer_kwargs", {}),
        ),
        _hrnet_handlers(),
    ),
    "hrnet_gru": lambda cfg: (HRNetConvGRU(cfg), _hrnet_gru_handlers()),
}


def build_model(cfg: DictConfig | dict[str, Any]):
    """Build a model instance and its IO handlers from config."""
    model_cfg = cfg["model"] if "model" in cfg else cfg
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)
    model_name = model_cfg.get("name")
    if model_name not in __factory:
        raise KeyError(f"invalid model: {model_name}")

    if model_name in ("hrnet", "hrcnet", "hrnet_gru"):
        return __factory[model_name](model_cfg)

    raise KeyError(f"Unsupported model: {model_name}")


__all__ = [
    "ClipSegmenter",
    "RuleBasedClipSegmenter",
    "TrajectoryCompleter",
    "PhysicsInterpolator",
    "BiLSTMCompleter",
    "HybridCompleter",
    "CompletionResult",
    "create_completer",
    "build_model",
    "HRNetConvGRU",
]
