"""Free-flight integration from a recorded state to output frames."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.physics.ball.dynamics import BallField, semi_implicit_euler_step


def integrate_flight(
    position: Tensor,
    velocity: Tensor,
    spin: Tensor,
    field: BallField,
    *,
    dt: float,
    substeps: int,
    frames: int,
) -> tuple[Tensor, Tensor]:
    """Integrate free flight and sample every ``substeps`` simulation steps.

    Args:
        position, velocity, spin: Initial state ``(..., 3)`` at output frame 0.
        field: Field parameters broadcastable to the state's leading shape.
        dt: Simulation step in seconds.
        substeps: Simulation steps per output frame.
        frames: Number of output frames to return, including the initial frame.

    Returns:
        Positions and velocities ``(..., frames, 3)``.  Frame 0 is the input state.
    """
    if substeps < 1 or frames < 1 or not dt > 0:
        raise ValueError("Require positive dt, substeps and frames")
    positions, velocities = [position], [velocity]
    for _ in range(frames - 1):
        for _ in range(substeps):
            position, velocity = semi_implicit_euler_step(
                position, velocity, spin, field, dt
            )
        positions.append(position)
        velocities.append(velocity)
    return torch.stack(positions, dim=-2), torch.stack(velocities, dim=-2)
