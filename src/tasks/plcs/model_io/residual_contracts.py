"""Model-facing contracts for PLCS residual geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class GeometryInput:
    features: NDArray[np.float32]
    init_world_m: NDArray[np.float32]
    raw_init_world_m: NDArray[np.float64]
    init_valid: NDArray[np.bool_]
    root_init_m: NDArray[np.float32]
    relative_init_m: NDArray[np.float32]
    time_positions: NDArray[np.float32]
    view_valid: NDArray[np.bool_]
    observations_uv: NDArray[np.float32]
    scores: NDArray[np.float32]
    reprojected_uv: NDArray[np.float32]
    residual_uv: NDArray[np.float32]
    used_views: NDArray[np.bool_]
    reprojection_valid: NDArray[np.bool_]
    ray_angle_deg: NDArray[np.float32]


def feature_dimension(joints: int) -> int:
    return 18 * joints + 61


def validate_model_inputs(
    features: Tensor, view_valid: Tensor, time_positions: Tensor, *, input_dim: int
) -> None:
    if features.ndim != 4 or features.shape[-1] != input_dim:
        raise ValueError("Expected features [B,V,T,F] for this residual profile")
    batch, _, frames, _ = features.shape
    if (
        view_valid.shape != features.shape[:3]
        or view_valid.dtype != torch.bool
        or time_positions.shape != (batch, frames)
    ):
        raise ValueError("Invalid residual view/time contract")
    if features.device != view_valid.device or features.device != time_positions.device:
        raise ValueError("Inputs must share a device")
