"""Output head modules for PLCS models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PositionHead(nn.Module):
    """Predict 3D position from latent representation."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        hidden_dim = int(hidden_dim)

        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Predict position from features."""
        return self.mlp(x)


class RotationHead(nn.Module):
    """Predict (cos(yaw), sin(yaw)) from latent representation."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        hidden_dim = int(hidden_dim)

        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Predict unit-normalized (cos, sin)."""
        out = self.mlp(x)
        return torch.nn.functional.normalize(out, dim=-1)
