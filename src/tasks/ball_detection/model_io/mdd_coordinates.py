"""Common tensor contracts for MDD coordinate detectors."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.ball_detection.models.mdd_pose.config import MDDPoseConfig


def validate_rgb_timestamps(config: MDDPoseConfig, rgb: Tensor, timestamps: Tensor) -> None:
    if rgb.ndim != 5 or rgb.shape[0] < 1 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("Native MDD models require RGB uint8 B,T,3,H,W")
    b, t, _, h, w = rgb.shape
    if min(h, w) < 8 or t != config.frames:
        raise ValueError("RGB requires 32 frames and spatial sizes at least eight")
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
