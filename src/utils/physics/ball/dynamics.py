"""Batched, differentiable in-flight tennis-ball dynamics.

The force model is the single source of truth for every consumer that simulates,
integrates or fits a ball trajectory::

    a = (0, 0, -g) - k_drag * |v - w| * (v - w) + k_magnus * omega x (v - w)

``w`` is the wind velocity.  Drag and Magnus act on the wind-relative velocity.
Spin is constant during free flight.  Steps use semi-implicit Euler, which is the
integrator of the dataset simulator, so integrating recorded states reproduces the
recorded trajectory.

All functions broadcast over leading dimensions: vectors are ``(..., 3)`` and
scalar parameters are tensors broadcastable to ``(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Below this relative speed the drag direction is undefined and drag is zero.
DRAG_SPEED_EPSILON = 1e-6


@dataclass(frozen=True)
class BallField:
    """Field parameters shared by every flight segment of one rally.

    ``k_drag`` and ``k_magnus`` are ``(...)`` tensors, ``wind`` is ``(..., 3)``.
    Disabled terms are expressed by a zero coefficient, never by a flag.
    """

    gravity: float
    k_drag: Tensor
    k_magnus: Tensor
    wind: Tensor

    def __post_init__(self) -> None:
        if not self.gravity > 0:
            raise ValueError(f"gravity must be positive, got {self.gravity}")
        if self.wind.shape[-1:] != (3,):
            raise ValueError(f"wind must end with 3 components, got {self.wind.shape}")


def acceleration(velocity: Tensor, spin: Tensor, field: BallField) -> Tensor:
    """Acceleration ``(..., 3)`` for velocity and spin ``(..., 3)``."""
    gravity = torch.zeros_like(velocity)
    gravity[..., 2] = -field.gravity
    relative = velocity - field.wind
    speed = relative.norm(dim=-1, keepdim=True)
    drag = -field.k_drag[..., None] * speed * relative
    drag = torch.where(speed > DRAG_SPEED_EPSILON, drag, torch.zeros_like(drag))
    magnus = field.k_magnus[..., None] * torch.linalg.cross(spin, relative, dim=-1)
    result: Tensor = gravity + drag + magnus
    return result


def semi_implicit_euler_step(
    position: Tensor, velocity: Tensor, spin: Tensor, field: BallField, dt: float
) -> tuple[Tensor, Tensor]:
    """One free-flight step: ``v += a(v) dt`` then ``p += v dt``."""
    velocity = velocity + acceleration(velocity, spin, field) * dt
    return position + velocity * dt, velocity
