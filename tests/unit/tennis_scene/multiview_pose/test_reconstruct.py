from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.assignment import assign
from src.tennis_scene.multiview_pose.inputs import PoseInputs
from src.tennis_scene.multiview_pose.reconstruct import reconstruct
from src.tennis_scene.multiview_pose.schema import RowRejection
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.utils.geometry.triangulation import PointRejection
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import _camera


def person() -> NDArray[np.float64]:
    joints: NDArray[np.float64] = np.array(
        [
            [0.0, 0.0, 1.68],
            [-0.03, 0.0, 1.70],
            [0.03, 0.0, 1.70],
            [-0.09, 0.02, 1.69],
            [0.09, 0.02, 1.69],
            [-0.19, 0.0, 1.49],
            [0.19, 0.0, 1.49],
            [-0.39, -0.06, 1.30],
            [0.36, 0.06, 1.31],
            [-0.54, -0.14, 1.17],
            [0.52, 0.09, 1.22],
            [-0.14, 0.0, 1.02],
            [0.14, 0.0, 1.02],
            [-0.17, 0.06, 0.58],
            [0.15, -0.08, 0.54],
            [-0.18, 0.10, 0.12],
            [0.18, -0.13, 0.10],
        ]
    )
    return joints


def cameras():
    return tuple(
        _camera(f"cam{i}", center)
        for i, center in enumerate(
            ([-8.0, -16.0, 9.0], [8.0, -16.0, 9.0], [5.0, 16.0, 9.0])
        )
    )


def observed_inputs(cameras, frames_by_view) -> PoseInputs:
    detections, poses, videos = [], [], []
    for camera, frames in zip(cameras, frames_by_view, strict=True):
        xyz = np.concatenate(frames, axis=0)
        uv, front = camera.project(xyz)
        assert front.all()
        assert np.all((uv >= 0) & (uv < [1280, 720]))
        boxes = np.concatenate((uv.min(axis=1) - 4, uv.max(axis=1) + 4), axis=1).astype(
            np.float32
        )
        offsets = np.cumsum([0, *map(len, frames)], dtype=np.int64)
        rows: NDArray[np.int64] = np.arange(len(uv), dtype=np.int64)
        scores: NDArray[np.float32] = np.full(len(uv), 0.9, np.float32)
        keypoints = np.concatenate(
            (uv, np.full(uv.shape[:-1] + (1,), 0.92)), axis=-1
        ).astype(np.float32)
        videos.append(
            SourceVideo(
                camera.camera_id,
                Path(f"{camera.camera_id}.mp4"),
                "a" * 64,
                len(frames),
                30.0,
                1280,
                720,
            )
        )
        detections.append(
            PersonDetectionOutput(camera.camera_id, offsets, boxes, scores, rows)
        )
        poses.append(
            PersonPoseFeaturesOutput(
                camera.camera_id, offsets, rows, keypoints, "b" * 64, "c" * 64
            )
        )
    return PoseInputs(
        ClipSource("video_000/clip_000", tuple(videos)),
        cameras,
        tuple(detections),
        tuple(poses),
    )


def test_sparse_native_timeline_reconstructs_all_people_with_stable_source_rows():
    first, second = person() + [0.0, -3.0, 0.0], person() + [6.0, 0.7, 0.0]
    group = np.stack((first, second))
    empty = np.zeros((0, 17, 3))
    inputs = observed_inputs(cameras(), [[group, empty, group[:1]]] * 3)
    result = reconstruct(inputs)
    np.testing.assert_array_equal(result.frame_offsets, [0, 2, 2, 3])
    np.testing.assert_array_equal(result.source_rows, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    np.testing.assert_allclose(
        result.positions_m, np.stack((first, second, first)), atol=1e-5
    )
    assert result.joint_valid.all() and result.inlier_mask.all()
    assert result.reprojection_px.max() < 1e-4
    for row_codes in result.row_rejection_codes:
        np.testing.assert_array_equal(row_codes, [RowRejection.ASSIGNED] * 3)

    duplicate = result.source_rows.copy()
    duplicate[1, 0] = duplicate[0, 0]
    with pytest.raises(ValueError, match="source row is reused"):
        replace(result, source_rows=duplicate)
    wrong_frame = result.source_rows.copy()
    wrong_frame[-1, 1] = 0
    with pytest.raises(ValueError, match="belongs to another frame"):
        replace(result, source_rows=wrong_frame)


def test_mixed_arm_in_one_view_rejects_that_row_and_preserves_independent_support():
    first, second = person() + [-1.2, -3.0, 0.0], person() + [2.3, 3.2, 0.0]
    people = np.stack((first, second))
    mixed = people.copy()
    mixed[0, [7, 9]] = second[[7, 9]]
    result = reconstruct(observed_inputs(cameras(), [[people], [mixed], [people]]))
    matches = {
        tuple(rows): pts
        for rows, pts in zip(result.source_rows, result.positions_m, strict=True)
    }
    assert set(matches) == {(0, -1, 0), (1, 1, 1)}
    np.testing.assert_allclose(matches[0, -1, 0], first, atol=1e-5)
    assert result.row_rejection_codes[1][0] == RowRejection.MIXED_POSE
    bad_view = int(np.flatnonzero(result.source_rows[:, 1] < 0)[0])
    assert not result.reprojection_valid[bad_view, 1].any()


def test_agreeing_wrong_arm_is_rejected_by_bone_length_even_with_low_pixel_error():
    first = person() + [-1.2, -3.0, 0.0]
    second = person() + [2.3, -3.0, 0.0]
    mix = first.copy()
    mix[[7, 9]] = second[[7, 9]]
    result = reconstruct(observed_inputs(cameras(), [[mix[None]]] * 3))
    np.testing.assert_array_equal(result.frame_offsets, [0, 0])
    for code in result.row_rejection_codes:
        np.testing.assert_array_equal(code, [RowRejection.MIXED_POSE])


def test_joint_dropout_keeps_other_joints_and_marks_the_missing_joint():
    xyz = person()[None] + [0.0, -3.0, 0.0]
    inputs = observed_inputs(cameras(), [[xyz]] * 3)
    weak = []
    for pose in inputs.poses[:2]:
        keypoints = pose.keypoints.copy()
        keypoints[0, 9, 2] = 0.0
        weak.append(replace(pose, keypoints=keypoints))
    result = reconstruct(replace(inputs, poses=(*weak, inputs.poses[2])))
    np.testing.assert_array_equal(result.source_rows, [[0, 0, 0]])
    assert result.joint_valid.sum() == 16 and not result.joint_valid[0, 9]
    assert result.joint_rejection_codes[0, 9] == PointRejection.INSUFFICIENT_VIEWS
    np.testing.assert_array_equal(result.positions_m[0, 9], np.zeros(3))
    assert not result.reprojection_valid[0, :, 9].any()


def test_single_view_and_parallel_rays_have_distinct_diagnostics():
    camera = cameras()[0]
    parallel = replace(camera, camera_id="cam1")
    xyz = person()[None] + [0.0, -3.0, 0.0]
    empty = np.zeros((0, 17, 3))
    result = reconstruct(
        observed_inputs((camera, parallel), [[xyz, xyz], [empty, xyz]])
    )
    np.testing.assert_array_equal(result.frame_offsets, [0, 0, 0])
    np.testing.assert_array_equal(
        result.row_rejection_codes[0],
        [RowRejection.INSUFFICIENT_VIEWS, RowRejection.DEGENERATE_RAYS],
    )
    np.testing.assert_array_equal(
        result.row_rejection_codes[1], [RowRejection.DEGENERATE_RAYS]
    )


def test_milp_finds_global_group_optimum_and_records_close_competition():
    groups = np.array(
        [[0, 0, -1], [0, 1, -1], [1, 0, -1], [2, -1, 0], [2, -1, 1], [-1, 2, 2]],
        np.int64,
    )
    scores = np.array([1.0, 0.7, 0.7, 1.0, 0.98, 0.9])
    selected = assign(groups, scores, 0.05)
    np.testing.assert_array_equal(selected.selected, [1, 2, 3, 5])
    np.testing.assert_array_equal(selected.competing, [True, True, True, False])
    np.testing.assert_array_equal(selected.ambiguous, [False, False, True, False])
    np.testing.assert_allclose(
        selected.relative_gap, [(1.4 - 1) / 1.4, (1.4 - 1) / 1.4, 0.02, 0.0]
    )
