"""Common tensor contracts for MDD coordinate detectors."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.ball_detection.models.mdd_pose.config import MDDPoseConfig


def validate_mdd_timestamps(config: MDDPoseConfig, mdd: Tensor, timestamps: Tensor) -> None:
    if mdd.ndim != 5 or mdd.shape[0] < 1 or mdd.shape[1] != 2:
        raise ValueError("MDD requires B,2,T,H,W")
    b, _, t, h, w = mdd.shape
    if min(h, w) < 8 or t != config.frames:
        raise ValueError("MDD requires 32 frames and spatial sizes at least eight")
    if timestamps.shape != (b, t) or timestamps.device != mdd.device:
        raise ValueError("Timestamps must align with MDD batch, frames and device")
    if any(x.dtype != torch.float32 or not bool(torch.isfinite(x).all()) for x in (mdd, timestamps)):
        raise ValueError("MDD and timestamps require finite float32 values")
    if bool(((mdd < 0) | (mdd > 1)).any()):
        raise ValueError("MDD sigmoid features must be in [0,1]")
    if bool((timestamps.diff(dim=1) <= 0).any()):
        raise ValueError("Real timestamps must increase strictly")


def decode_coordinates(config: MDDPoseConfig, output: Tensor) -> Tensor:
    if output.ndim != 3 or output.shape[1:] != (config.frames, 2) or not bool(torch.isfinite(output).all()):
        raise ValueError("Invalid per-frame coordinate output")
    return output
