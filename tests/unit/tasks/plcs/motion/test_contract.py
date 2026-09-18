"""Contract, persistence, and rigid-placement tests for PLCS motions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.plcs.motion import (
    Coco17MotionClip,
    MotionSourceKind,
    load_motion_clip,
    place_motion_on_court,
    save_motion_clip,
)
from src.utils.schema.court_normalization import denormalize_court_position


def _clip(*, frames: int = 4, fps: float = 30.0) -> Coco17MotionClip:
    root: NDArray[np.float32] = np.zeros((frames, 3), dtype=np.float32)
    root[:, 0] = np.arange(frames, dtype=np.float32)
    joints = np.repeat(root[:, None], 17, axis=1)
    joints[:, 5, 1] += 0.5
    return Coco17MotionClip(
        source_id="fixture-motion",
        source_path="/data/fixture.mp4",
        source_kind=MotionSourceKind.GVHMR,
        category="tennis",
        gender="neutral",
        fps=fps,
        timestamps_s=np.arange(frames, dtype=np.float64) / fps,
        joints_3d_m=joints,
        root_translation_m=root,
        root_rotation=np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
        joint_confidence=np.full((frames, 17), 0.75, dtype=np.float32),
        frame_valid=np.asarray([True] * (frames - 1) + [False], dtype=np.bool_),
        provenance={"camera_id": "cam1", "nested": {"track_id": 4}},
    )


def test_motion_artifact_round_trip_is_pickle_free_and_exact(tmp_path: Path) -> None:
    original = _clip()
    path = save_motion_clip(original, tmp_path / "fixture.motion.npz")

    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype != np.dtype(object) for key in archive.files)
    loaded = load_motion_clip(path)

    assert loaded.metadata() == original.metadata()
    np.testing.assert_array_equal(loaded.timestamps_s, original.timestamps_s)
    np.testing.assert_array_equal(loaded.joints_3d_m, original.joints_3d_m)
    np.testing.assert_array_equal(
        loaded.root_translation_m, original.root_translation_m
    )
    np.testing.assert_array_equal(loaded.root_rotation, original.root_rotation)
    np.testing.assert_array_equal(loaded.joint_confidence, original.joint_confidence)
    np.testing.assert_array_equal(loaded.frame_valid, original.frame_valid)
    assert not loaded.joints_3d_m.flags.writeable


def test_motion_contract_rejects_non_rotation_and_wrong_confidence_dtype() -> None:
    clip = _clip()
    invalid_rotation = clip.root_rotation.copy()
    invalid_rotation[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        replace(clip, root_rotation=invalid_rotation)
    with pytest.raises(TypeError, match="joint_confidence must use float32"):
        Coco17MotionClip(
            source_id=clip.source_id,
            source_path=clip.source_path,
            source_kind=clip.source_kind,
            category=clip.category,
            gender=clip.gender,
            fps=clip.fps,
            timestamps_s=clip.timestamps_s,
            joints_3d_m=clip.joints_3d_m,
            root_translation_m=clip.root_translation_m,
            root_rotation=clip.root_rotation,
            joint_confidence=clip.joint_confidence.astype(np.float64),
            frame_valid=clip.frame_valid,
            provenance=clip.provenance,
        )


def test_motion_contract_rejects_timestamps_that_disagree_with_fps() -> None:
    clip = _clip()
    timestamps = clip.timestamps_s.copy()
    timestamps[-1] += 0.01

    with pytest.raises(ValueError, match="constant interval 1 / fps"):
        replace(clip, timestamps_s=timestamps)


def test_court_placement_preserves_relative_translation_and_root_heading() -> None:
    clip = _clip()
    placed = place_motion_on_court(
        clip,
        initial_x_m=2.0,
        initial_y_m=-3.0,
        initial_yaw_rad=np.pi / 2,
    )

    world_root = denormalize_court_position(placed.position)
    np.testing.assert_allclose(world_root[0], [2.0, -3.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(world_root[-1], [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        placed.rotation,
        np.repeat(np.asarray([[0.0, 1.0]], dtype=np.float32), 4, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(placed.canonical_pose_3d[:, 5, 1], 0.5, atol=1e-6)
