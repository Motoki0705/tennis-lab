"""Free-flight values and derivatives from the discrete gravity-only solution."""

from __future__ import annotations

import pytest
import torch

from src.utils.physics.ball.dynamics import BallField
from src.utils.physics.ball.integrate import integrate_flight


@pytest.mark.parametrize(
    ("dt", "substeps", "frames"),
    [(0.125, 1, 5), (0.025, 4, 6), (0.2, 3, 1)],
    ids=["one_step_per_frame", "four_steps_per_frame", "frame_zero_only"],
)
def test_gravity_only_matches_discrete_closed_form(
    dt: float, substeps: int, frames: int
) -> None:
    position = torch.tensor(
        [[[1.0, -2.0, 3.0], [-4.0, 5.0, 2.0]], [[-2.0, 7.0, 4.0], [3.0, -1.0, 6.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    velocity = torch.tensor(
        [[[3.0, 4.0, 5.0], [2.0, -3.0, -1.0]], [[-4.0, 1.0, 0.0], [1.0, -2.0, 3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    field = BallField(
        9.8, position.new_tensor(0), position.new_tensor(0), torch.zeros_like(position)
    )
    positions, velocities = integrate_flight(
        position,
        velocity,
        torch.zeros_like(position),
        field,
        dt=dt,
        substeps=substeps,
        frames=frames,
    )
    # Sum the arithmetic progression v_1,...,v_n analytically. The integrator's
    # dt is the INTERNAL step width: output frame k has n = k * substeps.
    n = torch.arange(frames, dtype=position.dtype).unsqueeze(-1) * substeps
    a = position.new_tensor([0, 0, -9.8])
    expected_v = velocity.detach().unsqueeze(-2) + n * dt * a
    expected_p = (
        position.detach().unsqueeze(-2)
        + n * dt * velocity.detach().unsqueeze(-2)
        + a * dt**2 * n * (n + 1) / 2
    )
    torch.testing.assert_close(velocities, expected_v, atol=1e-12, rtol=0)
    torch.testing.assert_close(positions, expected_p, atol=1e-12, rtol=0)
    torch.testing.assert_close(positions[..., 0, :], position, atol=0, rtol=0)
    torch.testing.assert_close(velocities[..., 0, :], velocity, atol=0, rtol=0)

    # d sum(p_k + v_k)/d p_0 = frames; d/d v_0 = frames + sum(k*substeps*dt).
    grad_p, grad_v = torch.autograd.grad(
        positions.sum() + velocities.sum(), (position, velocity)
    )
    torch.testing.assert_close(
        grad_p, torch.full_like(position, frames), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        grad_v,
        torch.full_like(velocity, frames + dt * substeps * frames * (frames - 1) / 2),
        atol=1e-12,
        rtol=0,
    )
