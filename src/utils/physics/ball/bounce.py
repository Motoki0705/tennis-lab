"""Spin-dependent ground bounce of a tennis ball (Cross slide/grip model).

Impulses are integrated over the impact on a rigid, infinitely massive court.
The normal force acts a distance ``D`` ahead of the centre of mass along the
pre-impact horizontal travel direction ``d`` (Cross 2003), which reduces the
spin a friction impulse alone would produce.

The contact point at ``r = (0, 0, -R)`` slides with the horizontal velocity
``u = v_xy + (omega x r)_xy = (v_x - R omega_y, v_y + R omega_x)``.  Per unit
mass, the normal impulse is ``P = (1 + e_y) |v_z|`` and a horizontal friction
impulse ``J`` changes::

    v_xy  by J
    omega by (J_y, -J_x, 0) / (alpha R) + D P (d_y, -d_x, 0) / (alpha R^2)
    u     by (1 + 1/alpha) J + D P d / (alpha R)

- Normal: ``v_z' = -e_y v_z``.
- Grip: the contact point rebounds with ``u' = -e_x u`` (Cross 2002), i.e.
  ``J = -((1 + e_x) u + D P d / (alpha R)) / (1 + 1/alpha)``.
- Slide: when the grip impulse exceeds ``mu P``, the ball slides throughout the
  impact and ``J = -mu P u / |u|``.

``e_x = 0`` and ``D = 0`` give Brody's slide-or-roll model.  Spin about the
vertical axis is unchanged by a level court.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.physics.ball.surfaces import BallProperties

# Below these speeds the slip / travel directions are undefined.
SPEED_EPSILON = 1e-9


def _unit(vector: Tensor) -> tuple[Tensor, Tensor]:
    """Unit vectors and norms of ``(..., 2)``; zero where the norm vanishes."""
    norm = vector.norm(dim=-1)
    moving = norm > SPEED_EPSILON
    safe = torch.where(moving, norm, torch.ones_like(norm))
    return torch.where(moving[..., None], vector / safe[..., None], 0.0), norm


def ground_bounce(
    velocity: Tensor,
    spin: Tensor,
    *,
    restitution: Tensor | float,
    friction: Tensor | float,
    grip_restitution: Tensor | float,
    ball: BallProperties,
) -> tuple[Tensor, Tensor]:
    """Velocity and spin ``(..., 3)`` right after an impact with ``v_z < 0``.

    Surface coefficients are floats or tensors broadcastable to ``(...)``.
    """
    radius, alpha = ball.radius_m, ball.inertia_factor
    offset = ball.normal_force_offset_m
    restitution = torch.as_tensor(restitution, dtype=velocity.dtype)
    friction = torch.as_tensor(friction, dtype=velocity.dtype)
    grip_restitution = torch.as_tensor(grip_restitution, dtype=velocity.dtype)

    slip = torch.stack(
        (
            velocity[..., 0] - radius * spin[..., 1],
            velocity[..., 1] + radius * spin[..., 0],
        ),
        dim=-1,
    )
    slip_direction, _ = _unit(slip)
    travel, _ = _unit(velocity[..., :2])
    normal_impulse = (1 + restitution) * velocity[..., 2].abs()
    offset_slip = (offset * normal_impulse / (alpha * radius))[..., None] * travel

    grip = -((1 + grip_restitution)[..., None] * slip + offset_slip) / (1 + 1 / alpha)
    limit = friction * normal_impulse
    slide = -limit[..., None] * slip_direction
    gripping = grip.norm(dim=-1) <= limit
    impulse = torch.where(gripping[..., None], grip, slide)

    response = torch.cat(
        (velocity[..., :2] + impulse, (-restitution * velocity[..., 2])[..., None]),
        dim=-1,
    )
    torque = (
        impulse / (alpha * radius)
        + (offset * normal_impulse / (alpha * radius**2))[..., None] * travel
    )
    spin_change = torch.stack(
        (torque[..., 1], -torque[..., 0], torch.zeros_like(torque[..., 0])), dim=-1
    )
    return response, spin + spin_change
