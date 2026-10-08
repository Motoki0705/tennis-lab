"""Shared ball force model, free-flight integration and differentiability."""

from __future__ import annotations

import pytest
import torch

from src.utils.physics.ball import (
    BallField,
    acceleration,
    integrate_flight,
    semi_implicit_euler_step,
)


def _field(
    k_drag: float = 0.0,
    k_magnus: float = 0.0,
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dtype: torch.dtype = torch.float32,
) -> BallField:
    return BallField(
        gravity=9.8,
        k_drag=torch.tensor(k_drag, dtype=dtype),
        k_magnus=torch.tensor(k_magnus, dtype=dtype),
        wind=torch.tensor(wind, dtype=dtype),
    )


def test_gravity_only_acceleration_is_constant_downward() -> None:
    result = acceleration(torch.tensor([20.0, -3.0, 4.0]), torch.zeros(3), _field())
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, -9.8]))


def test_drag_opposes_wind_relative_velocity_with_quadratic_magnitude() -> None:
    velocity = torch.tensor([10.0, 0.0, 0.0])
    field = _field(k_drag=0.01, wind=(4.0, 0.0, 0.0))
    result = acceleration(velocity, torch.zeros(3), field)
    # Relative speed 6 m/s: drag = -0.01 * 6 * 6 along +x.
    torch.testing.assert_close(result, torch.tensor([-0.36, 0.0, -9.8]))


def test_drag_vanishes_when_moving_with_the_wind() -> None:
    field = _field(k_drag=0.02, wind=(3.0, -1.0, 0.0))
    result = acceleration(torch.tensor([3.0, -1.0, 0.0]), torch.zeros(3), field)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, -9.8]))


def test_topspin_magnus_pushes_a_forward_ball_down() -> None:
    # Ball travelling +y with spin about -x: omega x v = (0, 0, -|w||v|).
    velocity = torch.tensor([0.0, 25.0, 0.0])
    spin = torch.tensor([-200.0, 0.0, 0.0])
    result = acceleration(velocity, spin, _field(k_magnus=0.001))
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, -9.8 - 5.0]))


def test_integrate_flight_matches_explicit_steps_bitwise() -> None:
    field = _field(k_drag=0.013, k_magnus=0.0011, wind=(1.2, -0.7, 0.0))
    position = torch.tensor([0.5, -11.0, 1.1])
    velocity = torch.tensor([-1.0, 28.0, 4.5])
    spin = torch.tensor([-150.0, -60.0, 40.0])
    positions, velocities = integrate_flight(
        position, velocity, spin, field, dt=1 / 240, substeps=4, frames=6
    )
    p, v = position, velocity
    expected_p, expected_v = [p], [v]
    for _ in range(5):
        for _ in range(4):
            p, v = semi_implicit_euler_step(p, v, spin, field, 1 / 240)
        expected_p.append(p)
        expected_v.append(v)
    assert torch.equal(positions, torch.stack(expected_p))
    assert torch.equal(velocities, torch.stack(expected_v))


def test_batched_integration_equals_per_sample_integration() -> None:
    generator = torch.Generator().manual_seed(3)
    position = torch.randn(2, 3, 3, generator=generator)
    velocity = torch.randn(2, 3, 3, generator=generator) * 10
    spin = torch.randn(2, 3, 3, generator=generator) * 100
    k_drag = torch.rand(2, 3, generator=generator) * 0.02
    k_magnus = torch.rand(2, 3, generator=generator) * 0.002
    wind = torch.randn(2, 3, 3, generator=generator)
    batched, _ = integrate_flight(
        position,
        velocity,
        spin,
        BallField(9.8, k_drag, k_magnus, wind),
        dt=1 / 240,
        substeps=2,
        frames=4,
    )
    for i in range(2):
        for j in range(3):
            single, _ = integrate_flight(
                position[i, j],
                velocity[i, j],
                spin[i, j],
                BallField(9.8, k_drag[i, j], k_magnus[i, j], wind[i, j]),
                dt=1 / 240,
                substeps=2,
                frames=4,
            )
            torch.testing.assert_close(batched[i, j], single, rtol=1e-6, atol=1e-6)


def test_integration_is_differentiable_in_every_parameter() -> None:
    dtype = torch.float64
    inputs = (
        torch.tensor([0.2, -10.0, 1.0], dtype=dtype, requires_grad=True),
        torch.tensor([1.0, 25.0, 5.0], dtype=dtype, requires_grad=True),
        torch.tensor([-120.0, -50.0, 30.0], dtype=dtype, requires_grad=True),
        torch.tensor(0.012, dtype=dtype, requires_grad=True),
        torch.tensor(0.0009, dtype=dtype, requires_grad=True),
        torch.tensor([1.0, -2.0, 0.0], dtype=dtype, requires_grad=True),
    )

    def endpoint(
        position: torch.Tensor,
        velocity: torch.Tensor,
        spin: torch.Tensor,
        k_drag: torch.Tensor,
        k_magnus: torch.Tensor,
        wind: torch.Tensor,
    ) -> torch.Tensor:
        field = BallField(9.8, k_drag, k_magnus, wind)
        positions, _ = integrate_flight(
            position, velocity, spin, field, dt=1 / 240, substeps=3, frames=3
        )
        return positions[-1]

    assert torch.autograd.gradcheck(endpoint, inputs)


@pytest.mark.parametrize("bad", [{"dt": 0.0}, {"substeps": 0}, {"frames": 0}])
def test_integrate_flight_rejects_invalid_sampling(bad: dict[str, float]) -> None:
    arguments = {"dt": 1 / 240, "substeps": 1, "frames": 1} | bad
    with pytest.raises(ValueError):
        integrate_flight(
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            _field(),
            dt=float(arguments["dt"]),
            substeps=int(arguments["substeps"]),
            frames=int(arguments["frames"]),
        )
