"""PLCS generation tests for versioned position-only normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSequence
from src.tasks.plcs.generate_dataset.scene_generator import SceneGenerator
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


def _motion() -> MotionSequence:
    frames = 3
    trans = np.array(
        [[1.0, 2.0, 0.8], [1.5, 2.25, 0.9], [2.0, 2.5, 1.0]],
        dtype=np.float32,
    )
    joints: np.ndarray = np.zeros((frames, 52, 3), dtype=np.float32)
    joint_offsets = np.linspace(-0.5, 0.5, 52 * 3, dtype=np.float32).reshape(52, 3)
    joints[:] = trans[:, None, :] + joint_offsets[None, :, :]
    return MotionSequence(
        source_path="fixture.npz",
        category="test",
        gender="neutral",
        fps=30.0,
        poses=np.zeros((frames, 156), dtype=np.float32),
        trans=trans,
        betas=np.zeros(10, dtype=np.float32),
        joints_3d=joints,
    )


def _generator(version: str) -> SceneGenerator:
    generator = object.__new__(SceneGenerator)
    generator.court_coordinate_normalization = (
        resolve_court_coordinate_normalization(version)
    )
    return generator


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_generated_position_round_trips_to_physical_world_translation(
    version: str,
) -> None:
    generator = _generator(version)
    normalized, _rotation, canonical = generator._transform_motion_to_court(
        _motion(),
        init_x=2.0,
        init_y=-4.0,
        init_yaw=0.0,
    )

    world = generator.court_coordinate_normalization.denormalize_position(normalized)
    expected_world = np.array(
        [[2.0, -4.0, 0.8], [2.5, -3.75, 0.9], [3.0, -3.5, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(world, expected_world, atol=1.0e-5, rtol=0.0)
    np.testing.assert_allclose(
        canonical,
        np.broadcast_to(canonical[0:1], canonical.shape),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_v1_v2_generation_changes_only_normalized_translation_contract() -> None:
    generated = {}
    for version in ("v1", "v2"):
        generator = _generator(version)
        position, rotation, canonical = generator._transform_motion_to_court(
            _motion(),
            init_x=-1.5,
            init_y=3.0,
            init_yaw=0.4,
        )
        generated[version] = (position, rotation, canonical)

    v1_position, v1_rotation, v1_canonical = generated["v1"]
    v2_position, v2_rotation, v2_canonical = generated["v2"]
    np.testing.assert_array_equal(v1_rotation, v2_rotation)
    np.testing.assert_array_equal(v1_canonical, v2_canonical)
    np.testing.assert_allclose(
        resolve_court_coordinate_normalization("v1").denormalize_position(v1_position),
        resolve_court_coordinate_normalization("v2").denormalize_position(v2_position),
        atol=1.0e-5,
        rtol=0.0,
    )
