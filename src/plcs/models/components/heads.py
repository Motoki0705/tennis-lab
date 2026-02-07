"""Output head modules for PLCS.

These modules decode latent representations into position and rotation outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PositionHead(nn.Module):
    """Predict 3D position from latent representation.

    Outputs normalized (x, y, z) coordinates in court coordinate system.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the position head.

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
        """Predict position from features.

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            Tensor: Predicted position, shape (batch, 3).

        """
        return self.mlp(x)


class RotationHead(nn.Module):
    """Predict rotation (yaw) from latent representation.

    Outputs (sin(yaw), cos(yaw)) which can be converted to angle.
    Using sin/cos representation ensures continuity and avoids
    the discontinuity at +/- pi.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the rotation head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
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

        layers.append(nn.Linear(hidden_dim, 2))  # sin, cos
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Predict rotation from features.

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            Tensor: Predicted (sin, cos) of yaw, shape (batch, 2).
                The output is normalized to lie on the unit circle.

        """
        out = self.mlp(x)
        # Normalize to unit circle
        return torch.nn.functional.normalize(out, dim=-1)


class CombinedHead(nn.Module):
    """Combined head predicting both position and rotation.

    This can be more efficient than separate heads when features
    are highly correlated.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the combined head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.

        """
        super().__init__()

        # Shared layers
        shared_layers: list[nn.Module] = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            shared_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()

        # Separate output layers
        self.position_out = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.rotation_out = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict position and rotation from features.

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            tuple: (position (batch, 3), rotation (batch, 2)).

        """
        shared_feat = self.shared(x)

        position = self.position_out(shared_feat)
        rotation = self.rotation_out(shared_feat)
        rotation = torch.nn.functional.normalize(rotation, dim=-1)

        return position, rotation


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
