"""Ball-detector model implementations."""

from __future__ import annotations

from src.tasks.ball_detection.models.conv_next_unet import ConvNeXtUNet
from src.tasks.ball_detection.models.dinov3_rope import DINOv3RoPEBallDetector
from src.tasks.ball_detection.models.discriminators import (
    build_ball_detection_discriminator,
)
from src.tasks.ball_detection.models.spatiotemporal_unet import SpatioTemporalUNet

__all__ = [
    "ConvNeXtUNet",
    "DINOv3RoPEBallDetector",
    "SpatioTemporalUNet",
    "build_ball_detection_discriminator",
]
