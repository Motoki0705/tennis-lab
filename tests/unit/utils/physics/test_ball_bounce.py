"""Spin-dependent slide/grip bounce and the fixed surface table."""

from __future__ import annotations

import math

import pytest
import torch

from src.utils.physics.ball import (
    SURFACE_NAMES,
    SURFACES,
    TENNIS_BALL,
    BallProperties,
    ground_bounce,
    surface_properties,
)

RIGID = BallProperties(radius_m=0.033, inertia_factor=0.55, normal_force_offset_m=0.0)


def _bounce(
    velocity: list[float],
    spin: list[float],
    *,
    restitution: float = 0.8,
    friction: float = 0.6,
    grip_restitution: float = 0.15,
    ball: BallProperties = RIGID,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ground_bounce(
        torch.tensor(velocity, dtype=torch.float64),
        torch.tensor(spin, dtype=torch.float64),
        restitution=restitution,
        friction=friction,
        grip_restitution=grip_restitution,
        ball=ball,
    )


def _slip(velocity: torch.Tensor, spin: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.stack((velocity[0] - radius * spin[1], velocity[1] + radius * spin[0]))


def test_frictionless_court_only_reverses_the_vertical_velocity() -> None:
    velocity, spin = _bounce([3.0, 20.0, -8.0], [-100.0, 40.0, 25.0], friction=0.0)
    torch.testing.assert_close(
        velocity, torch.tensor([3.0, 20.0, 6.4], dtype=torch.float64)
    )
    torch.testing.assert_close(
        spin, torch.tensor([-100.0, 40.0, 25.0], dtype=torch.float64)
    )


def test_low_angle_bounce_slides_with_brody_speed_loss() -> None:
    # 10 degrees, no spin: friction mu (1 + e) |v_z| is below the grip impulse.
    velocity, spin = _bounce([0.0, 30.0, -5.0], [0.0, 0.0, 0.0])
    assert velocity[1].item() == pytest.approx(30.0 - 0.6 * 1.8 * 5.0)
    # Friction slows the bottom of the ball: topspin about -x for +y travel.
    assert spin[0].item() == pytest.approx(-(0.6 * 1.8 * 5.0) / (0.55 * 0.033))


def test_steep_bounce_grips_and_reverses_the_contact_point() -> None:
    velocity_in = torch.tensor([0.0, 10.0, -12.0], dtype=torch.float64)
    velocity, spin = _bounce(velocity_in.tolist(), [0.0, 0.0, 0.0], friction=0.9)
    before = _slip(velocity_in, torch.zeros(3, dtype=torch.float64), 0.033)
    after = _slip(velocity, spin, 0.033)
    torch.testing.assert_close(after, -0.15 * before)


def test_topspin_bounces_faster_and_flatter_than_backspin() -> None:
    incoming = [0.0, 22.0, -9.0]
    top_v, _ = _bounce(incoming, [-250.0, 0.0, 0.0], ball=TENNIS_BALL)
    back_v, _ = _bounce(incoming, [250.0, 0.0, 0.0], ball=TENNIS_BALL)
    assert top_v[1] > back_v[1]
    assert math.atan2(top_v[2], top_v[1]) < math.atan2(back_v[2], back_v[1])


def test_vertical_axis_spin_is_unchanged() -> None:
    _, spin = _bounce([5.0, 15.0, -7.0], [-80.0, 30.0, 140.0], ball=TENNIS_BALL)
    assert spin[2].item() == pytest.approx(140.0)


def test_normal_force_offset_reduces_the_acquired_topspin() -> None:
    _, rigid = _bounce([0.0, 29.1, -8.92], [0.0, 0.0, 0.0])
    _, offset = _bounce([0.0, 29.1, -8.92], [0.0, 0.0, 0.0], ball=TENNIS_BALL)
    assert rigid[0] < offset[0] < 0


def test_grass_bounce_matches_cross_high_speed_measurement() -> None:
    # Cross (2010) Fig. 3: 30.4 m/s at 17 deg on grass, COR 0.80, COF 0.53.
    velocity, _ = _bounce(
        [0.0, 29.1, -8.92], [0.0, 0.0, 0.0], restitution=0.80, friction=0.53
    )
    assert velocity[1].item() == pytest.approx(20.5, abs=0.2)
    assert velocity[2].item() == pytest.approx(7.14, abs=0.01)


def test_kinetic_energy_never_increases_for_a_rigid_contact() -> None:
    generator = torch.Generator().manual_seed(11)
    alpha, radius = RIGID.inertia_factor, RIGID.radius_m
    for _ in range(200):
        velocity = torch.randn(3, generator=generator, dtype=torch.float64) * 15
        velocity[2] = -velocity[2].abs() - 0.5
        spin = torch.randn(3, generator=generator, dtype=torch.float64) * 200
        surface = SURFACES[SURFACE_NAMES[int(torch.randint(3, (1,)).item())]]
        after_v, after_w = ground_bounce(
            velocity,
            spin,
            restitution=surface.restitution,
            friction=surface.friction,
            grip_restitution=surface.grip_restitution,
            ball=RIGID,
        )

        def energy(v: torch.Tensor, w: torch.Tensor) -> float:
            return float(v.square().sum() + alpha * radius**2 * w.square().sum())

        assert energy(after_v, after_w) <= energy(velocity, spin) + 1e-9


def test_batched_bounce_with_tensor_coefficients_is_differentiable() -> None:
    velocity = torch.tensor(
        [[0.0, 25.0, -9.0], [3.0, -12.0, -6.0]], dtype=torch.float64
    ).requires_grad_()
    spin = torch.tensor(
        [[-200.0, 10.0, 5.0], [100.0, -50.0, 0.0]], dtype=torch.float64
    ).requires_grad_()
    restitution = torch.tensor([0.8, 0.9], dtype=torch.float64, requires_grad=True)
    friction = torch.tensor([0.6, 0.8], dtype=torch.float64, requires_grad=True)

    def response(
        v: torch.Tensor, w: torch.Tensor, e: torch.Tensor, mu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ground_bounce(
            v, w, restitution=e, friction=mu, grip_restitution=0.15, ball=TENNIS_BALL
        )

    assert torch.autograd.gradcheck(response, (velocity, spin, restitution, friction))


def _court_pace_rating(name: str) -> float:
    surface = surface_properties(name)
    return 100 * (1 - surface.friction) + 150 * (0.81 - surface.restitution)


def test_surfaces_fall_in_their_itf_pace_categories() -> None:
    assert _court_pace_rating("clay") <= 29
    assert 35 <= _court_pace_rating("hard") <= 39
    assert _court_pace_rating("grass") >= 45


def test_unknown_surface_is_rejected() -> None:
    with pytest.raises(KeyError):
        surface_properties("carpet")
