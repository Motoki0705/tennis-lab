"""Prediction heads for role/existence/3D outputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


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


class BBoxHead(nn.Module):
    """Lightweight DETR-style head that predicts bbox center/size and class logits."""

    def __init__(self, dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.cls = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.exist = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.center = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        self.size = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        logits = self.cls(x)
        exist = torch.sigmoid(self.exist(x))
        center = self.center(x)
        size = F.softplus(self.size(x)) + 1e-3  # sizes must remain positive
        return {
            "cls_logits": logits,
            "exist_conf": exist,
            "bbox_center": center,
            "bbox_size": size,
        }
