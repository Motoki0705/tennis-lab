"""PLCS translation normalization and canonical-pose invariants."""

from __future__ import annotations

import numpy as np

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSequence
from src.tasks.plcs.generate_dataset.scene_generator import SceneGenerator
from src.utils.schema.court_normalization import denormalize_court_position


def test_transform_normalizes_only_court_translation() -> None:
    frames = 3
    trans = np.asarray(
        [[2.0, -3.0, 1.0], [3.0, -1.0, 1.1], [4.0, 1.0, 1.2]],
        dtype=np.float32,
    )
    joints: np.ndarray = np.zeros((frames, 52, 3), dtype=np.float32)
    joints[:, 0] = trans
    joints[:, 1] = trans
    joints[:, 1, 0] += 0.25
    motion = MotionSequence(
        source_path="test",
        category="test",
        gender="neutral",
        fps=30.0,
        poses=np.zeros((frames, 156), dtype=np.float32),
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        joints_3d=joints,
    )
    generator = object.__new__(SceneGenerator)

    position, rotation, canonical = generator._transform_motion_to_court(
        motion,
        init_x=1.5,
        init_y=-2.5,
        init_yaw=0.0,
    )

    expected_world = np.asarray(
        [[1.5, -2.5, 1.0], [2.5, -0.5, 1.1], [3.5, 1.5, 1.2]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        denormalize_court_position(position), expected_world, atol=1e-5, rtol=0.0
    )
    np.testing.assert_array_equal(rotation, np.tile([1.0, 0.0], (frames, 1)))
    np.testing.assert_allclose(canonical[:, 1, 0], 0.25, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(canonical[:, 1, 1:], 0.0, atol=0.0, rtol=0.0)
