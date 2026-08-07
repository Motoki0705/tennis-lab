"""Output head modules for PLCS models."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from src.utils.models.heads import MLPHead


class PositionHead(MLPHead):
    """Predict 3D position from latent representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict position from features."""
        return cast("Tensor", self.mlp(x))


class RotationHead(MLPHead):
    """Predict (cos(yaw), sin(yaw)) from latent representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict unit-normalized (cos, sin)."""
        out = self.mlp(x)
        return torch.nn.functional.normalize(out, dim=-1)


class CanonicalPoseHead(MLPHead):
    """Predict canonical 3D player joints from latent representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_keypoints: int,
    ) -> None:
        n_kp = int(num_keypoints)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_kp * 3,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_keypoints = n_kp

    def forward(self, x: Tensor) -> Tensor:
        """Predict canonical joints with shape ``(..., K, 3)``."""
        out = cast("Tensor", self.mlp(x))
        return out.reshape(*x.shape[:-1], self.num_keypoints, 3)
