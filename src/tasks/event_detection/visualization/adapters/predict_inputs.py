"""Model input adapters for event detection visualization prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.tasks.event_detection.visualization.types import Traj3DEventInputs, UVEventInputs


@dataclass(frozen=True)
class UVPredictInputs:
    """Tensorized predictor inputs for UV event models."""

    ball_uv: Tensor
    court_kp: Tensor
    ball_vis: Tensor
    court_vis: Tensor


@dataclass(frozen=True)
class Traj3DPredictInputs:
    """Tensorized predictor inputs for 3D event models."""

    ball_pos_world: Tensor


def build_uv_predict_inputs(inputs: UVEventInputs) -> UVPredictInputs:
    """Convert UV visualization inputs into predictor tensors."""
    return UVPredictInputs(
        ball_uv=torch.from_numpy(inputs.ball_uv).float(),
        court_kp=torch.from_numpy(inputs.court_kp).float(),
        ball_vis=torch.from_numpy(inputs.ball_vis.astype(np.float32)),
        court_vis=torch.from_numpy(inputs.court_vis.astype(np.float32)),
    )


def build_traj3d_predict_inputs(inputs: Traj3DEventInputs) -> Traj3DPredictInputs:
    """Convert 3D visualization inputs into predictor tensors."""
    return Traj3DPredictInputs(
        ball_pos_world=torch.from_numpy(inputs.ball_pos_world).float(),
    )
