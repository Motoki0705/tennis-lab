from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import FramePoseCandidates, RowRejection
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.visualization.pose_comparison.inputs import (
    AdoptedScene,
    ClipPlan,
    ComparisonInputs,
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
    draw_poses,
    draw_rejections,
)
from src.tennis_scene.visualization.pose_comparison.report import (
    clip_metrics,
    write_page,
)
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


def test_page_embeds_literal_data_for_file_playback(tmp_path: Path) -> None:
    write_page(tmp_path, {"stages": [{"label": "</script>"}]})
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '"label": "\\u003c/script>"' in html
    assert '<script id="comparison-data" type="application/json">{' in html
    assert (tmp_path / "manifest.json").is_file()
