"""Ground-bounce response of a tennis ball."""

from __future__ import annotations

from torch import Tensor


def ground_bounce(
    velocity: Tensor, spin: Tensor, *, restitution: float, friction: float
) -> tuple[Tensor, Tensor]:
    """Velocity and spin ``(..., 3)`` right after a ground impact.

    The vertical component reverses with ``restitution``; the horizontal
    components keep ``1 - friction``.  Spin is unchanged.
    """
    response = velocity.clone()
    response[..., 2] = -restitution * velocity[..., 2]
    response[..., :2] = (1 - friction) * velocity[..., :2]
    return response, spin.clone()
