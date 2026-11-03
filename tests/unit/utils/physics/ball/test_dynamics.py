"""Absolute semi-implicit Euler values independent of the integration helper."""

from __future__ import annotations

import torch

from src.utils.physics.ball.dynamics import BallField, semi_implicit_euler_step


def test_gravity_only_step_uses_updated_velocity() -> None:
    # h=0.25 s, a_z=-8 m/s^2: v_z becomes 4, so p_z becomes 8 (not 8.5).
    position = torch.tensor([2.0, -3.0, 7.0], dtype=torch.float64)
    velocity = torch.tensor([4.0, -2.0, 6.0], dtype=torch.float64)
    zero = position.new_tensor(0.0)
    field = BallField(8.0, zero, zero, torch.zeros(3, dtype=position.dtype))
    actual_p, actual_v = semi_implicit_euler_step(
        position, velocity, torch.zeros_like(position), field, dt=0.25
    )
    torch.testing.assert_close(
        actual_v, position.new_tensor([4, -2, 4]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual_p, position.new_tensor([3, -3.5, 8]), atol=0, rtol=0
    )
