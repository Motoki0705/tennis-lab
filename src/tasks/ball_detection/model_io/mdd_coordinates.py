"""Common tensor contracts for MDD coordinate detectors."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.ball_detection.models.mdd.config import MDDPoseConfig, MDDPretrainConfig


def validate_rgb_input(rgb: Tensor, *, frames: int) -> None:
    """Validate decoded RGB at the caller boundary, before entering forward."""
    if rgb.ndim != 5 or rgb.shape[0] < 1 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("Native MDD models require RGB uint8 B,T,3,H,W")
    if min(rgb.shape[-2:]) < 8 or rgb.shape[1] != frames:
        raise ValueError(f"RGB requires {frames} frames and spatial sizes at least eight")


def validate_rgb_timestamps(config: MDDPoseConfig | MDDPretrainConfig, rgb: Tensor, timestamps: Tensor) -> None:
    validate_rgb_input(rgb, frames=config.frames)
    b, t = rgb.shape[:2]
    if timestamps.shape != (b, t) or timestamps.device != rgb.device:
        raise ValueError("Timestamps must align with RGB batch, frames and device")
    if timestamps.dtype != torch.float32 or not bool(torch.isfinite(timestamps).all()):
        raise ValueError("Timestamps require finite float32 values")
    if bool((timestamps.diff(dim=1) <= 0).any()):
        raise ValueError("Real timestamps must increase strictly")


def decode_coordinates(config: MDDPoseConfig, output: Tensor) -> Tensor:
    if output.ndim != 3 or output.shape[1:] != (config.frames, 2) or not bool(torch.isfinite(output).all()):
        raise ValueError("Invalid per-frame coordinate output")
    return output
