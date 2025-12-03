"""Output head modules for BLCS.

These modules decode latent representations into 3D trajectory outputs.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Trajectory3DHead(nn.Module):
    """Predict 3D positions from sequence features.

    Outputs normalized (x, y, z) coordinates in court coordinate system
    for each frame in the sequence.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the trajectory head.

        Args:
            input_dim: Input feature dimension.
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
        """Predict 3D positions from features.

        Args:
            x: Input features, shape (B, T, input_dim).

        Returns:
            Tensor: Predicted positions, shape (B, T, 3).

        """
        return self.mlp(x)


class VelocityHead(nn.Module):
    """Predict 3D velocities from sequence features.

    Optional head for velocity supervision during training.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the velocity head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (3 for vx, vy, vz).
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
        """Predict 3D velocities from features.

        Args:
            x: Input features, shape (B, T, input_dim).

        Returns:
            Tensor: Predicted velocities, shape (B, T, 3).

        """
        return self.mlp(x)
