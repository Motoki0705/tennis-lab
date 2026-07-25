"""Unit tests for PLCS visualization prediction output selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.tasks.plcs.visualization.api.predict import _apply_canonical_pose_source


def _scene(canonical_pose: np.ndarray | None) -> SimpleNamespace:
    return SimpleNamespace(canonical_pose_3d=canonical_pose)


def test_gt_source_preserves_scene_canonical_pose() -> None:
    gt_pose: np.ndarray = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    predicted_pose = np.full_like(gt_pose, 99.0)
    scene = _scene(gt_pose.copy())

    _apply_canonical_pose_source(scene, predicted_pose, "gt")

    np.testing.assert_array_equal(scene.canonical_pose_3d, gt_pose)


def test_prediction_source_replaces_scene_canonical_pose() -> None:
    gt_pose: np.ndarray = np.zeros((2, 3, 3), dtype=np.float32)
    predicted_pose: np.ndarray = np.ones((2, 4, 3), dtype=np.float32)
    scene = _scene(gt_pose)

    _apply_canonical_pose_source(scene, predicted_pose, "prediction")

    np.testing.assert_array_equal(scene.canonical_pose_3d, predicted_pose)
    assert scene.canonical_pose_3d is not predicted_pose


def test_prediction_source_requires_model_canonical_pose() -> None:
    scene = _scene(np.zeros((2, 3, 3), dtype=np.float32))

    with pytest.raises(
        ValueError,
        match="requires a model that outputs canonical_pose",
    ):
        _apply_canonical_pose_source(scene, None, "prediction")


def test_prediction_source_rejects_frame_count_mismatch() -> None:
    scene = _scene(np.zeros((2, 3, 3), dtype=np.float32))
    predicted_pose: np.ndarray = np.zeros((3, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="frame count must match"):
        _apply_canonical_pose_source(scene, predicted_pose, "prediction")


def test_prediction_source_rejects_non_xyz_pose() -> None:
    scene = _scene(np.zeros((2, 3, 3), dtype=np.float32))
    predicted_pose: np.ndarray = np.zeros((2, 4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"must have shape \(T, J, 3\)"):
        _apply_canonical_pose_source(scene, predicted_pose, "prediction")
