"""Preselected BLCS trajectory output implementations."""

from __future__ import annotations

from typing import TypeAlias

from torch import Tensor, nn

from src.utils.models.heads import MLPHead


class PositionTrajectoryOutput(nn.Module):
    """Decode trajectory features into a fixed position-only mapping."""

    def __init__(self, *, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.position = MLPHead(
            input_dim=input_dim,
            hidden_dim=input_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """Return the preselected position-only raw model output."""
        return {"position": self.position(features)}


class PositionVelocityTrajectoryOutput(nn.Module):
    """Decode trajectory features into a fixed position/velocity mapping."""

    def __init__(self, *, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.position = MLPHead(
            input_dim=input_dim,
            hidden_dim=input_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity = MLPHead(
            input_dim=input_dim,
            hidden_dim=input_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """Return the preselected position/velocity raw model output."""
        return {
            "position": self.position(features),
            "velocity": self.velocity(features),
        }


TrajectoryOutput: TypeAlias = (
    PositionTrajectoryOutput | PositionVelocityTrajectoryOutput
)


def build_trajectory_output(
    *, input_dim: int, dropout: float, predict_velocity: bool
) -> TrajectoryOutput:
    """Select one fixed trajectory decoder during model construction."""
    if predict_velocity:
        return PositionVelocityTrajectoryOutput(
            input_dim=input_dim,
            dropout=dropout,
        )
    return PositionTrajectoryOutput(input_dim=input_dim, dropout=dropout)


__all__ = [
    "PositionTrajectoryOutput",
    "PositionVelocityTrajectoryOutput",
    "build_trajectory_output",
]
