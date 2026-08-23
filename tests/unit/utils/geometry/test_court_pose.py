"""Unit tests for :mod:`src.utils.geometry.court_pose`."""

from __future__ import annotations

import math

import pytest
import torch

from src.utils.geometry.court_pose import (
    canonical_pose_to_world_pose,
    court_position_to_world_translation,
    world_pose_to_canonical_pose,
)
from src.utils.schema.court import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
)
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


class TestCourtPositionToWorldTranslation:
    def test_scales_each_axis(self) -> None:
        position = torch.tensor([1.0, 1.0, 1.0])
        out = court_position_to_world_translation(position)
        expected = torch.tensor(
            [COURT_COORD_SCALE_X, COURT_COORD_SCALE_Y, COURT_COORD_SCALE_Z]
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_zero_position_maps_to_origin(self) -> None:
        out = court_position_to_world_translation(torch.zeros(3))
        assert torch.allclose(out, torch.zeros(3))

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_selected_contract_controls_all_translation_axes(self, version: str) -> None:
        contract = resolve_court_coordinate_normalization(version)

        out = court_position_to_world_translation(
            torch.ones(2, 3),
            normalization=contract,
        )

        torch.testing.assert_close(
            out,
            torch.tensor(contract.scale_xyz).expand(2, -1),
        )


def _rotation(theta: float) -> torch.Tensor:
    return torch.tensor([math.cos(theta), math.sin(theta)])


class TestCanonicalWorldRoundTrip:
    def test_identity_placement_is_noop(self) -> None:
        canonical = torch.randn(5, 3)
        position = torch.zeros(3)
        rotation = _rotation(0.0)  # (cos 0, sin 0) = (1, 0)
        world = canonical_pose_to_world_pose(canonical, position, rotation)
        assert torch.allclose(world, canonical, atol=1e-5)

    def test_round_trip_recovers_canonical(self) -> None:
        canonical = torch.randn(8, 3)
        position = torch.tensor([0.3, -0.5, 0.2])
        rotation = _rotation(0.7)
        world = canonical_pose_to_world_pose(canonical, position, rotation)
        recovered = world_pose_to_canonical_pose(world, position, rotation)
        assert torch.allclose(recovered, canonical, atol=1e-4)

    def test_z_axis_is_pure_translation(self) -> None:
        canonical = torch.randn(4, 3)
        position = torch.tensor([0.1, 0.2, 1.0])
        rotation = _rotation(1.1)
        world = canonical_pose_to_world_pose(canonical, position, rotation)
        # Yaw only rotates in the XY plane; Z just shifts by the scaled offset.
        expected_z = canonical[..., 2] + position[2] * COURT_COORD_SCALE_Z
        assert torch.allclose(world[..., 2], expected_z, atol=1e-5)

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_canonical_meter_offsets_are_not_rescaled(self, version: str) -> None:
        canonical = torch.tensor(
            [[0.2, -0.4, 1.7], [-0.3, 0.5, 0.1]], dtype=torch.float64
        )
        position = torch.tensor([0.25, -0.5, 0.125], dtype=torch.float64)
        rotation = _rotation(0.0).to(torch.float64)
        contract = resolve_court_coordinate_normalization(version)

        world = canonical_pose_to_world_pose(
            canonical,
            position,
            rotation,
            normalization=contract,
        )
        translation = torch.tensor(contract.scale_xyz, dtype=torch.float64) * position

        torch.testing.assert_close(world - translation, canonical)
        torch.testing.assert_close(
            world_pose_to_canonical_pose(
                world,
                position,
                rotation,
                normalization=contract,
            ),
            canonical,
        )
