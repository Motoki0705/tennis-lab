"""Prediction heads for role/existence/3D outputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PredictionHead(nn.Module):
    """Maps decoder states to structured outputs."""

    def __init__(self, dim: int, smpl_param_dim: int) -> None:
        super().__init__()
        self.role = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        self.exist = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.ball_xyz = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.player_xyz = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.smpl = (
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, smpl_param_dim))
            if smpl_param_dim > 0
            else _ZeroProjection()
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Convert decoder features into structured predictions."""
        return {
            "role_logits": self.role(x),
            "exist_conf": torch.sigmoid(self.exist(x)),
            "ball_xyz": self.ball_xyz(x),
            "player_xyz": self.player_xyz(x),
            "smpl": self.smpl(x),
        }


class _ZeroProjection(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1] + (0,)
        return x.new_zeros(shape)
