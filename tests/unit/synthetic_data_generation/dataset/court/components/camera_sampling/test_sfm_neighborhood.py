"""Tests for support-bounded court novel-view sampling."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_sampling.sfm_neighborhood import (
    NovelViewThresholds,
    pose_distance_score,
    sample_safe_novel_views,
)
from src.synthetic_data_generation.scene_contract import (
    SceneCamera,
    SimilarityTransform,
)
from src.utils.schema.court import court_keypoints_3d


def _look_at(
    center: tuple[float, float, float],
    target: tuple[float, float, float],
) -> NDArray[np.float64]:
    centre_array = np.asarray(center, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - centre_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack((right, down, forward))
    result[:3, 3] = centre_array
    return result


def _camera(camera_id: str, center_x: float = 0.0) -> SceneCamera:
    pose = _look_at((center_x, -30.0, 10.0), (0.0, 0.0, 0.0))
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="fixture",
        image_uri=f"{camera_id}.png",
        source_frame_index=int(camera_id.rsplit("_", maxsplit=1)[-1]),
        group_id=0,
        width=640,
        height=480,
        intrinsics=(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


def _support_points() -> NDArray[np.float64]:
    x, y = np.meshgrid(
        np.linspace(-12.0, 12.0, 20),
        np.linspace(-20.0, 20.0, 24),
    )
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def test_same_seed_is_exact_and_all_selected_views_pass_gates() -> None:
    cameras = (_camera("camera_0", -1.0), _camera("camera_1", 1.0))
    identity = SimilarityTransform(
        scale=1.0,
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(0.0, 0.0, 0.0),
    )
    kwargs = {
        "cameras": cameras,
        "court_from_scene": identity,
        "court_keypoints_court": court_keypoints_3d().numpy(),
        "support_points_scene": _support_points(),
        "seed": 17,
        "proposals_per_anchor": 32,
        "max_views": 8,
    }

    first = sample_safe_novel_views(**kwargs)
    second = sample_safe_novel_views(**kwargs)

    assert first == second
    assert first.safe_anchor_count == 2
    assert len(first.selected) == 8
    for camera in first.selected:
        assert all(camera.court_keypoints_visible[:14])
        assert camera.min_court_depth_m > 0.10
        assert camera.min_line_margin_px >= 0.0
        assert camera.collision_clearance_m >= 0.25
        assert camera.extrapolation_score <= 1.0 + 1.0e-10


def test_narrow_intrinsics_fail_without_relaxing_framing_gate() -> None:
    camera = _camera("camera_0")
    narrow = SceneCamera(
        camera_id=camera.camera_id,
        source_camera_id=camera.source_camera_id,
        image_uri=camera.image_uri,
        source_frame_index=camera.source_frame_index,
        group_id=camera.group_id,
        width=camera.width,
        height=camera.height,
        intrinsics=(
            3000.0,
            0.0,
            320.0,
            0.0,
            3000.0,
            240.0,
            0.0,
            0.0,
            1.0,
        ),
        camera_to_scene=camera.camera_to_scene,
    )
    identity = SimilarityTransform(
        scale=1.0,
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="No captured camera"):
        sample_safe_novel_views(
            (narrow,),
            identity,
            court_keypoints_3d().numpy(),
            _support_points(),
            seed=0,
            proposals_per_anchor=4,
            max_views=1,
        )


def test_pose_distance_uses_coupled_normalized_metric() -> None:
    limits = NovelViewThresholds()
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[0, 3] = limits.translation_limit_m

    assert pose_distance_score(first, second, limits) == pytest.approx(1.0)


def test_invalid_output_count_is_rejected() -> None:
    identity = SimilarityTransform(
        scale=1.0,
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="max_views"):
        sample_safe_novel_views(
            (_camera("camera_0"),),
            identity,
            court_keypoints_3d().numpy(),
            _support_points(),
            seed=0,
            proposals_per_anchor=4,
            max_views=0,
        )
