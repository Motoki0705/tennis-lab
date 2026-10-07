"""Ball-detector model implementations."""

from __future__ import annotations

from src.tasks.ball_detection.models.conv_next_unet import ConvNeXtUNet
from src.tasks.ball_detection.models.discriminators import (
    build_ball_detection_discriminator,
)
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDPoseDetector

__all__ = [
    "ConvNeXtUNet",
    "MDDPoseConfig",
    "MDDPoseDetector",
    "build_ball_detection_discriminator",
]
