"""Prediction heads for ball detector."""

from __future__ import annotations

import torch
from torch import nn


class XYHead(nn.Module):
    """Regresses normalized ball coordinates from latent feature vector."""

    def __init__(self, in_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(in_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisibilityHead(nn.Module):
    """Predicts visibility logit for each frame."""

    def __init__(self, in_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
