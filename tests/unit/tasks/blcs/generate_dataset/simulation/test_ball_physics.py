"""Fixed court normalization tests at the BLCS physics boundary."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.ball_physics import (
    BallPhysics,
    BallState,
    PhysicsConfig,
)


def test_position_normalization_uses_one_scale_and_round_trips() -> None:
    physics = object.__new__(BallPhysics)
    positions_m = torch.tensor(
        [[-5.485, -11.885, 0.0], [5.485, 11.885, 1.07]],
        dtype=torch.float32,
    )

    normalized = physics.normalize_position(positions_m)
    recovered = physics.denormalize_position(normalized)

    torch.testing.assert_close(
        normalized,
        positions_m / 11.885,
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(recovered, positions_m, atol=1e-5, rtol=0.0)


def _config(*, use_drag: bool, use_magnus: bool) -> PhysicsConfig:
    return PhysicsConfig(
        gravity=9.81,
        k_drag=0.013,
        k_magnus=0.0011,
        e_z=0.75,
        mu=0.1,
        alpha_net=0.3,
        alpha_net_cord=0.05,
        alpha_fence=0.3,
        net_half_thickness=0.03,
        net_cord_radius=0.03,
        dt=1 / 240,
        use_drag=use_drag,
        use_magnus=use_magnus,
        wind=(1.5, -0.5, 0.0),
        gravity_range=None,
        k_drag_range=None,
        k_magnus_range=None,
        e_z_range=None,
        mu_range=None,
        wind_speed_range=None,
        wind_direction_range_deg=None,
    )


def _reference_step(config: PhysicsConfig, state: BallState) -> BallState:
    """The scalar formulation used before the shared batched force model."""
    accel = torch.tensor([0.0, 0.0, -config.gravity])
    v_rel = state.velocity - torch.tensor(config.wind, dtype=torch.float32)
    if config.use_drag:
        speed_rel = v_rel.norm()
        if speed_rel > 1e-6:
            accel = accel + -config.k_drag * speed_rel * v_rel
    if config.use_magnus:
        accel = accel + config.k_magnus * torch.linalg.cross(state.spin, v_rel)
    velocity = state.velocity + accel * config.dt
    return BallState(
        position=state.position + velocity * config.dt,
        velocity=velocity,
        spin=state.spin.clone(),
    )


@pytest.mark.parametrize(
    ("use_drag", "use_magnus"), [(True, True), (True, False), (False, True)]
)
def test_shared_force_model_reproduces_scalar_simulator_bitwise(
    use_drag: bool, use_magnus: bool
) -> None:
    config = _config(use_drag=use_drag, use_magnus=use_magnus)
    physics = BallPhysics(config)
    state = BallState(
        position=torch.tensor([0.3, -11.5, 1.0]),
        velocity=torch.tensor([-1.0, 27.0, 5.0]),
        spin=torch.tensor([-170.0, -55.0, 80.0]),
    )
    reference = state.clone()
    for _ in range(300):
        state = physics.step(state)
        reference = _reference_step(config, reference)
        assert torch.equal(state.position, reference.position)
        assert torch.equal(state.velocity, reference.velocity)
