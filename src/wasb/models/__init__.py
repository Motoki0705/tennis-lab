"""Models for WASB tennis dataset generation."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from .clip_segmenter import ClipSegmenter, RuleBasedClipSegmenter
from .hrnet import HRNet
from .trajectory_completer import (
    BiLSTMCompleter,
    CompletionResult,
    HybridCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    create_completer,
)

__factory: dict[str, Any] = {
    "hrnet": HRNet,
}


def build_model(cfg: DictConfig | dict[str, Any]):
    """Build a model instance from config."""
    model_cfg = cfg["model"] if "model" in cfg else cfg
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)
    model_name = model_cfg.get("name")
    if model_name not in __factory:
        raise KeyError(f"invalid model: {model_name}")

    if model_name == "hrnet":
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
]
