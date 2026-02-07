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
    """Predict (sin(yaw), cos(yaw)) from latent representation."""

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
        """Predict unit-normalized (sin, cos)."""
        out = self.mlp(x)
        return torch.nn.functional.normalize(out, dim=-1)

class PerTokenKeypoint3DHead(nn.Module):
    """Predict per-token 3D keypoints from token features.

    Applies a shared MLP to each token independently.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize per-token 3D head.

        Args:
            input_dim: Input feature dimension per token.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (default 3 for x, y, z).
            num_layers: Number of hidden layers.
            dropout: Dropout probability.

        """
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Predict per-token 3D coordinates.

        Args:
            x: Input token features, shape (batch, num_tokens, input_dim).

        Returns:
            Tensor: Predicted 3D coordinates, shape (batch, num_tokens, 3).

        """
        return self.mlp(x)
