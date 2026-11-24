from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import FramePoseCandidates, RowRejection
from src.tennis_scene.multiview_tracking.schema import (
    NODE,
    SCHEMA,
    VERSION,
    PlayerDecision,
    PlayerSelection,
    PoseTracks,
    SampleEvent,
    TrackEnd,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.visualization.pose_comparison.inputs import (
    AdoptedScene,
    ClipPlan,
    ComparisonInputs,
    load_tracking,
)
from src.tennis_scene.visualization.pose_comparison.rendering import (
    BALL,
    GUTTER,
    HEADER_H,
    PALETTE,
    SIDE_W,
    VIEW_H,
    VIEW_W,
    FrameRenderer,
    Poses,
    TrackOverlay,
    draw_poses,
    draw_rejections,
)
from src.tennis_scene.visualization.pose_comparison.report import (
    clip_metrics,
    tracked_clip_metrics,
    write_page,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.geometry.triangulation import PinholeCamera, PointRejection


def test_unassigned_boxes_record_reason_codes_and_no_assigned_outline(
    monkeypatch,
) -> None:
    captions = []
    set_text = cv2.putText

    def capture_label(*args):
        captions.append(args[1])
        return set_text(*args)

    monkeypatch.setattr(cv2, "putText", capture_label)
    image: NDArray[np.uint8] = np.zeros((80, 130, 3), dtype=np.uint8)
    codes = np.array(list(RowRejection), dtype=np.uint8)
    boxes = np.array([(20 * i + 3, 30, 20 * i + 16, 66) for i in range(len(codes))])
    draw_rejections(image, boxes, codes)
    assert captions[::2] == ["V", "D", "G", "M", "C"]
    assert not image[:, :20].any()
    colors = {tuple(image[45, x]) for x in (23, 43, 63, 83, 103)}
    assert (0, 0, 0) not in colors and len(colors) == 5


def test_skeleton_drawing_distinguishes_close_assignment_and_masks_joint(
    monkeypatch,
) -> None:
    captions = []
    set_text = cv2.putText

    def capture_label(*args):
        captions.append(args[1])
        return set_text(*args)

    monkeypatch.setattr(cv2, "putText", capture_label)
    image: NDArray[np.uint8] = np.zeros((90, 150, 3), dtype=np.uint8)
    points: NDArray[np.float64] = np.zeros((2, 17, 2), dtype=np.float64)
    points[:, 5] = [[30, 45], [110, 45]]
    points[:, 6] = [[50, 45], [130, 45]]
    points[:, 7] = [[30, 70], [110, 70]]
    valid: NDArray[np.bool_] = np.zeros((2, 17), dtype=np.bool_)
    valid[:, 5:7] = True
    poses = Poses(
        np.zeros((2, 17, 3), dtype=np.float32),
        valid,
        (3, 1),
        "N",
        np.array([False, True]),
    )
    draw_poses(image, points, valid, poses)
    assert captions[::2] == ["N3", "N1", "?"]
    np.testing.assert_allclose(image[45, 30], PALETTE[3], atol=4)
    np.testing.assert_allclose(image[45, 110], PALETTE[1], atol=4)
    assert not image[64:, :].any()


def _comparison() -> tuple[ComparisonInputs, ClipPlan]:
    source = ClipSource(
        "video_000/clip_012",
        tuple(
            SourceVideo(f"cam{i}", Path(f"cam{i}.mp4"), str(i) * 64, 2, 60.0, 320, 180)
            for i in range(4)
        ),
    )
    cameras = tuple(
        PinholeCamera(
            v.camera_id,
            np.array([[100.0, 0, 160], [0, 100, 90], [0, 0, 1]]),
            np.eye(3),
            np.zeros(3),
        )
        for v in source.videos
    )
    xyz: NDArray[np.float32] = np.zeros((2, 17, 3), dtype=np.float32)
    xyz[:, 5:7] = [[[6, 0, 3], [6, 1, 3]], [[2, 3, 3], [2, 4, 3]]]
    valid: NDArray[np.bool_] = np.zeros((2, 17), dtype=np.bool_)
    valid[:, 5:7] = True
    support: NDArray[np.bool_] = np.zeros((2, 4, 17), dtype=np.bool_)
    support[:, :2, 5:7] = True
    candidates = FramePoseCandidates(
        camera_ids=source.camera_ids,
        frame_offsets=np.array([0, 2, 2]),
        positions_m=xyz,
        joint_valid=valid,
        source_rows=np.array([[0, 0, -1, -1], [1, 1, -1, -1]]),
        inlier_mask=support,
        reprojection_px=np.zeros(support.shape, dtype=np.float32),
        reprojection_valid=support,
        joint_rejection_codes=np.where(
            valid, 0, PointRejection.INSUFFICIENT_VIEWS
        ).astype(np.uint8),
        scores=np.ones(2, dtype=np.float32),
        ambiguous=np.array([False, True]),
        competing_assignment=np.array([False, True]),
        relative_score_gap=np.zeros(2, dtype=np.float32),
        detection_offsets=(
            np.array([0, 2, 2]),
            np.array([0, 2, 2]),
            np.zeros(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
        ),
        row_rejection_codes=(
            np.zeros(2, dtype=np.uint8),
            np.zeros(2, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
        ),
    )
    ball = np.array([[0.2, -0.3, 3], [0.2, -0.3, 3]], dtype=np.float32)
    old = AdoptedScene(
        cameras, (14,), xyz[None], valid[None], ball, np.array([True, False]), {}, None
    )
    data = ComparisonInputs(
        source,
        old,
        (),
        candidates,
        cameras,
        (np.zeros((2, 4), dtype=np.float32),) * 2
        + (np.empty((0, 4), dtype=np.float32),) * 2,
        {"legacy_scene_assembly": {"artifact_id": "a" * 64}},
    )
    return data, ClipPlan(source.clip_id, Path(source.clip_id), 2, None, None, 1.0)


def test_paired_camera_tiles_share_frame_and_ball_mask() -> None:
    data, plan = _comparison()
    renderer = FrameRenderer(data, plan)
    views: tuple[NDArray[np.uint8], ...] = tuple(
        np.full((180, 320, 3), 30 + 30 * i, dtype=np.uint8) for i in range(4)
    )
    hit = renderer.render(0, views)
    missing = renderer.render(1, views)
    for i in range(4):
        row, col = divmod(i, 2)
        y = HEADER_H + row * VIEW_H
        for x in (col * VIEW_W, SIDE_W + GUTTER + col * VIEW_W):
            assert np.array_equal(hit[y + 150, x + 300], views[i][150, 300])
            # The native (166⅔, 80) pixels are scaled to the 384×216 tile.
            assert tuple(hit[y + 96, x + 200]) == BALL
            assert tuple(missing[y + 96, x + 200]) != BALL


def test_frame_histograms_keep_empty_frames_and_outside_core_people() -> None:
    data, plan = _comparison()
    metrics = clip_metrics(data, plan, 2.0)
    assert metrics["new"]["populated_frames"] == 1
    assert metrics["new"]["count_histogram"] == {0: 1, 2: 1}
    assert metrics["new"]["core_count_histogram"] == {0: 1, 1: 1}
    assert metrics["new"]["ambiguous_frames"] == 1
    assert metrics["legacy"]["count_histogram"] == {1: 2}
    assert metrics["ball_valid_frames"] == 1


@pytest.fixture
def time_tracks() -> PoseTracks:
    # P0 is measured, occluded, then observed again. T1 leaves after frame 0.
    torso = np.array([[-2, -4, 3], [6, 2, 3]] + [[-2, -4, 3]] * 3, dtype=np.float32)
    xyz: NDArray[np.float32] = np.zeros((5, 17, 3), dtype=np.float32)
    xyz[:, 5] = torso + [-0.3, 0, 0]
    xyz[:, 6] = torso + [0.3, 0, 0]
    measured: NDArray[np.bool_] = np.zeros((5, 17), dtype=bool)
    measured[[0, 1, 3, 4], 5:7] = True
    predicted = np.zeros_like(measured)
    predicted[2, 5:7] = True
    return PoseTracks(
        camera_ids=("cam0", "cam1", "cam2", "cam3"),
        frame_offsets=np.array([0, 2, 3, 4, 5]),
        track_ids=np.array([0, 1, 0, 0, 0]),
        candidate_indices=np.array([0, 1, -1, 2, 3]),
        source_rows=np.array(
            [[0, 0, -1, -1], [1, 1, -1, -1], [-1] * 4, [2, 2, -1, -1], [3, 3, -1, -1]]
        ),
        positions_m=xyz,
        measured_joints=measured,
        predicted_joints=predicted,
        torso_m=torso,
        velocity_mps=np.zeros_like(torso),
        velocity_valid=np.zeros(5, dtype=bool),
        event=np.array(
            [
                SampleEvent.BORN,
                SampleEvent.BORN,
                SampleEvent.OCCLUDED,
                SampleEvent.REAPPEARED,
                SampleEvent.OBSERVED,
            ],
            dtype=np.uint8,
        ),
        frame_competing=np.array([True, False, False, False, False]),
        frame_gap=np.array([0.01, 0, 0, 0, 0], dtype=np.float32),
        frame_ambiguous=np.array([True, False, False, False, False]),
        temporal_margin=np.array([-1, -1, 0, 0, -1], dtype=np.float32),
        id_ambiguous=np.array([False, False, True, True, False]),
        end_frames=np.array([4, 1]),
        end_reasons=np.array(
            [TrackEnd.CLIP_BOUNDARY, TrackEnd.DISAPPEARED], dtype=np.uint8
        ),
        selection=PlayerSelection(
            1,
            1,
            (
                PlayerDecision(3, 3, (3, 0), 0, -1, 0.0, 0, 3.0, "selected"),
                PlayerDecision(
                    1, 0, (0, 0), 0, 0, 0.0, 0, 0.0, "insufficient_residence"
                ),
            ),
        ),
    )


def test_tracking_overlay_identifies_selected_track_through_short_occlusion(
    time_tracks: PoseTracks, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = TrackOverlay(time_tracks, fps=10.0)
    born, occluded, returned = [overlay.poses(frame) for frame in (0, 1, 2)]
    first, bystander = born.track_marks
    gap, again = occluded.track_marks[0], returned.track_marks[0]
    assert (first.label, bystander.label, gap.label, again.label) == (
        "P0+",
        "T1+",
        "P0~",
        "P0R",
    )
    assert first.selected and not bystander.selected and first.color != bystander.color
    assert first.color == gap.color == again.color
    assert first.ambiguity == "A?" and again.ambiguity == "A? ID?"

    uv: NDArray[np.float64] = np.zeros((1, 17, 2), dtype=np.float64)
    uv[0, 5:7] = [(20, 40), (65, 40)]
    dashed: NDArray[np.uint8] = np.zeros((60, 80, 3), dtype=np.uint8)
    solid = np.zeros_like(dashed)
    draw_poses(dashed, uv, occluded.valid, occluded)
    draw_poses(solid, uv, returned.valid, returned)
    np.testing.assert_allclose(dashed[40, 22], first.color, atol=4)
    assert dashed[40, 27].max() < solid[40, 27].max() / 2
    np.testing.assert_allclose(solid[40, 27], first.color, atol=4)

    captions: list[str] = []
    monkeypatch.setattr(cv2, "putText", lambda image, label, *_: captions.append(label))
    court: NDArray[np.uint8] = np.zeros((432, 170, 3), dtype=np.uint8)
    overlay.draw_disappearances(court, 0)
    assert not captions
    overlay.draw_disappearances(court, 1)
    overlay.draw_disappearances(court, 4)
    assert captions[::2] == ["T1X", "T1X"]
    np.testing.assert_allclose(court[200, 132], bystander.color, atol=4)


def test_stage2_coverage_excludes_interpolation_and_unselected_people(
    time_tracks: PoseTracks,
) -> None:
    stage1 = {
        "source_frames": 4,
        "legacy": {"populated_frames": 2},
        "new": {"reconstruction_seconds": 5.0},
    }
    metrics = tracked_clip_metrics(stage1, time_tracks, 1.5)
    current = metrics["new"]
    assert metrics["legacy"] == stage1["legacy"]
    assert current["count_histogram"] == {0: 1, 1: 3}
    assert current["populated_frames"] == 3
    assert current["interpolated_frames"] == 1
    assert current["selected_by_side"] == [[0], []]
    assert current["missing_per_side"] == (0, 1)
    assert current["status"] == "incomplete"
    assert current["temporal_ambiguous_frames"] == 2


def test_tracking_store_rejects_a_different_candidate_revision(
    tmp_path: Path, time_tracks: PoseTracks
) -> None:
    data, _ = _comparison()
    source = replace(
        data.source, videos=tuple(replace(v, num_frames=4) for v in data.source.videos)
    )
    store = ClipStore(tmp_path, json_value(source))
    candidate = {"artifact_id": "a" * 64, "sha256": "b" * 64}
    config = {"players_per_side": 1}
    ref = store.publish(
        NODE,
        time_tracks,
        ArtifactCodec(PoseTracks),
        schema=SCHEMA,
        version=VERSION,
        identity={"implementation": "test"},
        dependencies={},
        provenance={"frame_pose_candidates": candidate, "config": config},
    )
    record: dict[str, Any] = {
        "config": config,
        "clips": {
            source.clip_id: {
                "source_frames": 4,
                "candidate_artifact_id": candidate["artifact_id"],
                "candidate_descriptor_sha256": candidate["sha256"],
                "tracking_artifact_id": ref.artifact_id,
                "tracking_descriptor_sha256": ref.sha256,
            }
        },
    }
    data = replace(
        data,
        source=source,
        provenance={"source_sha256": store.source_key, "candidate_artifact": candidate},
    )
    reader = StoreReader(store.index_path)
    loaded, provenance = load_tracking(reader, data, record)
    assert loaded.selection.selected_track_ids == (0,)
    assert provenance["tracking_artifact"]["artifact_id"] == ref.artifact_id

    # Set the run to another candidate too. The saved derivation must still pin the original.
    record["clips"][source.clip_id]["candidate_descriptor_sha256"] = "c" * 64
    data.provenance["candidate_artifact"] = {**candidate, "sha256": "c" * 64}
    with pytest.raises(ValueError, match="Tracking inputs or settings disagree"):
        load_tracking(reader, data, record)


def test_page_embeds_literal_data_for_file_playback(tmp_path: Path) -> None:
    write_page(tmp_path, {"stages": [{"label": "</script>"}]})
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '"label": "\\u003c/script>"' in html
    assert '<script id="comparison-data" type="application/json">{' in html
    assert (tmp_path / "manifest.json").is_file()
