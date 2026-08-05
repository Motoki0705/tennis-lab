"""Model factory for ball detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.ball_detection.configuration import validate_model
from src.tasks.ball_detection.models.conv_next_unet import ConvNeXtUNet
from src.tasks.ball_detection.models.dinov3_rope import DINOv3RoPEBallDetector
from src.tasks.ball_detection.models.discriminators import (
    build_ball_detection_discriminator,
)
from src.tasks.ball_detection.models.input_adapter import (
    resolve_input_layout,
    resolve_input_mode,
    resolve_model_in_channels,
    to_model_input,
)
from src.tasks.ball_detection.models.spatiotemporal_unet import SpatioTemporalUNet

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_ball_detection_model(config: DictConfig) -> nn.Module:
    """Build a ball detection model from ``config.model.name``."""
    model_cfg = validate_model(config)
    model_name = str(model_cfg["name"])
    if model_name == "stunet":
        return SpatioTemporalUNet.from_config(config)
    if model_name == "conv_next_unet":
        return ConvNeXtUNet.from_config(config)
    if model_name == "dinov3_rope":
        return DINOv3RoPEBallDetector.from_config(config)
    raise ValueError(
        "Unknown ball_detection model.name="
        f"'{model_name}'. Supported: "
        "['stunet', 'conv_next_unet', 'dinov3_rope']"
    )


__all__ = [
    "ConvNeXtUNet",
    "DINOv3RoPEBallDetector",
    "SpatioTemporalUNet",
    "build_ball_detection_discriminator",
    "build_ball_detection_model",
    "resolve_input_mode",
    "resolve_input_layout",
    "resolve_model_in_channels",
    "to_model_input",
]
