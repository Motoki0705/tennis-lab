"""Normalization-contract tests for BLCS physical simulation outputs."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.ball_physics import (
    BallPhysics,
    BallState,
    PhysicsConfig,
)
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


def _physics(version: str) -> BallPhysics:
    return BallPhysics(
        PhysicsConfig(
            gravity=9.81,
            k_drag=0.0,
            k_magnus=0.0,
            e_z=0.7,
            mu=0.2,
            alpha_net=0.2,
            alpha_net_cord=0.3,
            alpha_fence=0.2,
            net_half_thickness=0.03,
            net_cord_radius=0.02,
            dt=1.0 / 30.0,
            use_drag=False,
            use_magnus=False,
            wind=(0.0, 0.0, 0.0),
            gravity_range=None,
            k_drag_range=None,
            k_magnus_range=None,
            e_z_range=None,
            mu_range=None,
            wind_speed_range=None,
            wind_direction_range_deg=None,
        ),
        normalization=version,
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_ball_physics_position_and_velocity_use_the_selected_contract(
    version: str,
) -> None:
    physics = _physics(version)
    contract = resolve_court_coordinate_normalization(version)
    position_m = torch.tensor([[5.485, 11.885, 1.07], [-2.0, 4.0, 8.0]])
    velocity_mps = torch.tensor([[9.0, -18.0, 4.5], [-1.0, 2.0, -3.0]])

    position_norm = physics.normalize_position(position_m)
    velocity_norm = physics.normalize_velocity(velocity_mps)

    torch.testing.assert_close(
        position_norm,
        position_m / torch.tensor(contract.scale_xyz),
    )
    torch.testing.assert_close(
        velocity_norm,
        velocity_mps / torch.tensor(contract.scale_xyz),
    )
    torch.testing.assert_close(physics.denormalize_position(position_norm), position_m)
    torch.testing.assert_close(physics.denormalize_velocity(velocity_norm), velocity_mps)


def test_v2_gravity_in_physics_remains_physical_meters_per_second_squared() -> None:
    physics = _physics("v2")
    zero = torch.zeros(3)
    acceleration = physics.compute_acceleration(
        BallState(position=zero, velocity=zero, spin=zero)
    )

    torch.testing.assert_close(acceleration, torch.tensor([0.0, 0.0, -9.81]))
