"""Model factory for ball detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from src.tasks.ball_detection.models.dino_pseudo3d import DINOPseudo3DBallDetector
from src.tasks.ball_detection.models.spatiotemporal_unet import SpatioTemporalUNet

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_ball_detection_model(config: "DictConfig") -> nn.Module:
    """Build a ball detection model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "stunet"))
    if model_name == "stunet":
        return SpatioTemporalUNet.from_config(config)
    if model_name == "dino_pseudo3d":
        return DINOPseudo3DBallDetector.from_config(config)
    raise ValueError(
        "Unknown ball_detection model.name="
        f"'{model_name}'. Supported: ['stunet', 'dino_pseudo3d']"
    )


__all__ = [
    "DINOPseudo3DBallDetector",
    "SpatioTemporalUNet",
    "build_ball_detection_model",
]
