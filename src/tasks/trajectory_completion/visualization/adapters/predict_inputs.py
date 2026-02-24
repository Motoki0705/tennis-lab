"""Model input adapters for trajectory completion visualization prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.tasks.trajectory_completion.visualization.types import TrajectoryInputs


@dataclass(frozen=True)
class UVCompletionPredictInputs:
    """Tensorized predictor inputs for UV trajectory completion models."""

    ball_uv: Tensor
    ball_vis: Tensor
    ball_mask: Tensor
    court_kp: Tensor
    court_vis: Tensor


def build_uv_completion_predict_inputs(
    inputs: TrajectoryInputs,
) -> UVCompletionPredictInputs:
    """Convert visualization inputs into completion predictor tensors."""
    t_len = int(inputs.ball_uv_in.shape[0])
    return UVCompletionPredictInputs(
        ball_uv=torch.from_numpy(inputs.ball_uv_in).float(),
        ball_vis=torch.from_numpy(inputs.ball_obs_mask.astype(np.float32)),
        ball_mask=torch.ones((t_len,), dtype=torch.float32),
        court_kp=torch.from_numpy(inputs.court_kp).float(),
        court_vis=torch.from_numpy(inputs.court_vis.astype(np.float32)),
    )
