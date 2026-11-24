"""Contracts at the saved tracking, scene assembly and publication boundaries."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.person_tracking.features import FeatureConfig
from src.tennis_scene.multiview_pose.schema import FramePoseCandidates, RowRejection
from src.tennis_scene.multiview_scene.inputs import assemble_players
from src.tennis_scene.multiview_scene.publication import require_known_pointers
from src.tennis_scene.multiview_scene.report import completion
from src.tennis_scene.multiview_tracking.schema import TrackingConfig
from src.tennis_scene.multiview_tracking.tracking import track
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.utils.geometry.triangulation import PointRejection


def test_crops_use_the_original_detector_row_per_camera_and_preserve_gap_masks() -> (
    None
):
    camera_ids = ("cam2", "cam0", "cam1")
    source = ClipSource(
        "test/clip0",
        tuple(
            SourceVideo(c, Path(f"{c}.mp4"), "a" * 64, 6, 30.0, 100, 100)
            for c in camera_ids
        ),
    )
    timeline = [(f, p) for f in range(6) for p in range(2) if (f, p) != (1, 0)]
    measurements: NDArray[np.float32] = np.zeros(
        (len(timeline), 17, 3), dtype=np.float32
    )
    valid: NDArray[np.bool_] = np.zeros((len(timeline), 17), dtype=bool)
    valid[:, [5, 6, 11, 12]] = True
    rows = np.array(
        [
            [2 * f + p, 2 * f + p if p == 1 else -1, 2 * f + p if p == 0 else -1]
            for f, p in timeline
        ],
        dtype=np.int64,
    )
    for i, (_, p) in enumerate(timeline):
        y = 5 if p else -4
        measurements[i, [5, 6, 11, 12]] = [
            [-0.2, y, 1.5],
            [0.2, y, 1.5],
            [-0.15, y, 1.0],
            [0.15, y, 1.0],
        ]
    support = (rows >= 0)[..., None] & valid[:, None]
    detector_offsets = np.arange(0, 13, 2)
    codes = []
    for values in rows.T:
        c: NDArray[np.uint8] = np.full(
            12, RowRejection.INSUFFICIENT_VIEWS, dtype=np.uint8
        )
        c[values[values >= 0]] = RowRejection.ASSIGNED
        codes.append(c)
    candidates = FramePoseCandidates(
        camera_ids=camera_ids,
        frame_offsets=np.array([0, 2, 3, 5, 7, 9, 11]),
        source_rows=rows,
        positions_m=measurements,
        joint_valid=valid,
        joint_rejection_codes=np.where(
            valid, PointRejection.VALID, PointRejection.INSUFFICIENT_VIEWS
        ).astype(np.uint8),
        inlier_mask=support,
        reprojection_px=np.zeros_like(support, dtype=np.float32),
        reprojection_valid=support,
        scores=np.ones(11, np.float32),
        ambiguous=np.zeros(11, bool),
        competing_assignment=np.zeros(11, bool),
        relative_score_gap=np.zeros(11, np.float32),
        detection_offsets=(detector_offsets,) * 3,
        row_rejection_codes=tuple(codes),
    )
    tracks = track(candidates, fps=source.fps, config=TrackingConfig(1))
    detection_rows = np.arange(12)
    boxes = np.array(
        [[10 + 3 * row, 20, 22 + 3 * row, 50] for row in range(12)], np.float32
    )
    detections = tuple(
        PersonDetectionOutput(
            c, detector_offsets, boxes, np.ones(12, np.float32), detection_rows
        )
        for c in camera_ids
    )
    poses = tuple(
        PersonPoseFeaturesOutput(
            c,
            detector_offsets,
            detection_rows,
            np.array(
                [[[5 + 12 * view + 5 * row, 60, 0.9]] * 17 for row in range(12)],
                np.float32,
            ),
            "a" * 64,
            "b" * 64,
        )
        for view, c in enumerate(camera_ids)
    )
    result = assemble_players(
        source,
        tracks,
        candidates,
        detections,
        poses,
        players_per_side=1,
        threshold=0.25,
        crops=FeatureConfig(),
    )
    assert result.sides == (-1, 1)
    np.testing.assert_array_equal(result.source_rows[0, :, 2], [4, -1, 4])
    np.testing.assert_array_equal(result.grouped.uv_px[0, 2, 2, 5], [49, 60])
    np.testing.assert_allclose(result.raw.boxes_xys[2, 2, 0], [28, 35, 36])
    np.testing.assert_array_equal(result.source_rows[0, :, 1], [-1, -1, -1])
    assert result.interpolated_joints[0, 1, 5] and not result.skeleton.valid[0, 1].any()
    assert not result.raw.observed[:, 1, 0].any()
    with pytest.raises(ValueError, match="Required 2 selected IDs per side"):
        assemble_players(
            source,
            tracks,
            candidates,
            detections,
            poses,
            players_per_side=2,
            threshold=0.25,
            crops=FeatureConfig(),
        )


def test_completion_distinguishes_body_availability_from_scene_and_ball() -> None:
    missing = completion("test/clip0", 5, (3, 8), {3: -1, 8: 1}, 1, None, {})
    assert missing["selected_negative"] == missing["selected_positive"] == 1
    assert "scene_missing" in missing["reason"] and not missing["complete"]
    fields: dict[str, Any] = {
        "player_track_ids": np.array([3, 8]),
        "player_valid": np.ones((2, 5), dtype=bool),
        "player_smpl_valid": np.array([[1, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=bool),
        "player_kp_3d_vis": np.zeros((2, 5, 17), dtype=bool),
        "ball_3d_valid": np.array([1, 1, 0, 0, 0], dtype=bool),
    }
    scene = completion("test/clip0", 5, (3, 8), {3: -1, 8: 1}, 1, None, fields)
    assert scene["all_placed_frames"] == 1
    assert scene["all_gvhmr_frames"] == 0
    assert scene["ball_3d_frames"] == 2 and scene["ball_3d_ratio"] == 0.4
    assert scene["reason"] == "gvhmr_missing_for_selected_id" and not scene["complete"]


def test_publication_resume_accepts_half_switch_and_rejects_concurrent_update() -> None:
    before = {"scene.json": "base scene", "annotation.json": "base annotation"}
    after = {"scene.json": "new scene", "annotation.json": "new annotation"}
    half = {**before, "scene.json": after["scene.json"]}
    require_known_pointers(half, before, after)
    with pytest.raises(ValueError, match="changed independently"):
        require_known_pointers(
            {**half, "annotation.json": "unrelated change"}, before, after
        )
