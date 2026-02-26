"""Adapters for building ball multitask predictor inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.developing.ball_multitask.visualization.types import SceneInputs


@dataclass(frozen=True)
class BallMultitaskPredictInputs:
    """Predictor-ready tensors for a single sequence."""

    ball_uv: torch.Tensor
    ball_vis: torch.Tensor
    court_kp: torch.Tensor
    court_vis: torch.Tensor
    seq_len: torch.Tensor


def build_ball_multitask_predict_inputs(inputs: SceneInputs) -> BallMultitaskPredictInputs:
    """Convert scene arrays into predictor input tensors."""
    return BallMultitaskPredictInputs(
        ball_uv=torch.from_numpy(inputs.ball_uv).float(),
        ball_vis=torch.from_numpy(inputs.ball_vis).float(),
        court_kp=torch.from_numpy(inputs.court_kp).float(),
        court_vis=torch.from_numpy(inputs.court_vis).float(),
        seq_len=torch.tensor(inputs.seq_len, dtype=torch.long),
    )
