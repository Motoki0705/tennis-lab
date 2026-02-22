"""Trajectory head adapter for multi-task learning."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.blcs.models.components.heads import Trajectory3DHead


class Trajectory3DHeadAdapter(nn.Module):
    """Predict 3D ball trajectories from sequence features.

    Wraps the BLCS Trajectory3DHead to keep configuration local to
    the multi-task model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim_value = int(hidden_dim) if hidden_dim is not None else int(input_dim // 2)
        self.head = Trajectory3DHead(
            input_dim=int(input_dim),
            hidden_dim=hidden_dim_value,
            output_dim=3,
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: Hidden states, shape (B, T, D).

        Returns:
            3D trajectory predictions, shape (B, T, 3).
        """
        return self.head(x)


if __name__ == "__main__":
    import torch

    head = Trajectory3DHeadAdapter(input_dim=64, dropout=0.0)
    x = torch.randn(2, 8, 64)
    y = head(x)
    assert y.shape == (2, 8, 3)
    print("trajectory_head smoke ok")
