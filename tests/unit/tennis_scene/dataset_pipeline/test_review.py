"""Review projection, unsupported-label handling, and camera-order checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.review import project_teacher, render_review
from src.tennis_scene.schema import SceneResult


def _scene() -> SceneResult:
    return SceneResult(
        num_frames=2, fps=30, width=100, height=100,
        court_kp=np.zeros((1, 2, 20, 2), np.float32), court_vis=np.ones((1, 2, 20), np.float32),
        player_position=np.zeros((2, 2, 3), np.float32), player_yaw=np.zeros((2, 2), np.float32),
        player_kp_3d=np.broadcast_to(np.array([2., 4., 2.], np.float32), (2, 2, 17, 3)).copy(),
        ball_3d=np.array([[1, 2, 2], [1, 2, -2]], np.float32),
        metadata={"reference": {"camera_fits": [{"R": np.eye(3).tolist(), "t": [0, 0, 0], "K": [[10, 0, 50], [0, 10, 50], [0, 0, 1]]}]},
                  "label_quality": {"player_weight": [[1, 1], [0, 0]], "ball_weight": [1, 1]}},
    )


def test_projection_respects_weights_and_positive_depth() -> None:
    scene = _scene()
    result = project_teacher(scene, 0, 0)
    np.testing.assert_allclose(result["pose"][0, 0], [60, 70])
    np.testing.assert_allclose(result["ball"][0], [55, 60])
    assert result["pose_valid"][0].all()
    assert not result["pose_valid"][1].any()
    assert result["ball_valid"][0]
    assert not project_teacher(scene, 0, 1)["ball_valid"][0]
    scene.metadata["label_quality"]["ball_weight"] = [0, 1]
    assert not project_teacher(scene, 0, 0)["ball_valid"][0]


def test_review_requires_explicit_quality_weights() -> None:
    scene = _scene()
    scene.metadata["label_quality"]["player_weight"] = [[float("nan")]*2]*2
    with pytest.raises(ValueError, match="finite"):
        project_teacher(scene, 0, 0)


def test_review_rejects_camera_order_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    scene = _scene()
    scene.metadata["reference"]["camera_ids"] = ["wrong_camera"]
    monkeypatch.setattr("src.tennis_scene.dataset_pipeline.review.ClipManifest.load", lambda _: SimpleNamespace(camera_ids=("cam0",)))
    monkeypatch.setattr("src.tennis_scene.dataset_pipeline.review.load_slcs_annotation", lambda _: scene)
    with pytest.raises(ValueError, match="camera order"):
        render_review(tmp_path, tmp_path / "output", frames=[0])


def test_render_persists_requested_frame_order_and_visibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace

    import cv2

    scene = _scene()
    scene.metadata["reference"]["camera_ids"] = ["cam0"]
    scene.human_kp_2d = np.full((2, 1, 2, 17, 2), 0.5, np.float32)
    scene.human_kp_vis = np.zeros((2, 1, 2, 17), np.float32)
    scene.human_kp_vis[0, 0, 1] = 1
    scene.ball_uv = np.full((1, 2, 2), 0.5, np.float32)
    scene.ball_vis = np.array([[False, True]])
    media = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(media), cv2.VideoWriter.fourcc(*"MJPG"), 30, (100, 100))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((100, 100, 3), np.uint8))
    writer.release()
    manifest = SimpleNamespace(camera_ids=("cam0",), clip_id="v/c", video_id="v", camera_index=lambda _: 0, media_path=lambda _: media)
    monkeypatch.setattr("src.tennis_scene.dataset_pipeline.review.ClipManifest.load", lambda _: manifest)
    monkeypatch.setattr("src.tennis_scene.dataset_pipeline.review.load_slcs_annotation", lambda _: scene)
    monkeypatch.setattr("src.tennis_scene.dataset_pipeline.review.sha256", lambda _: "fixture")
    image, sidecar = render_review(tmp_path, tmp_path / "review", frames=[1, 0], panel_width=320)
    decoded = cv2.imread(str(image))
    assert decoded is not None
    assert decoded.shape == (840, 320, 3)
    report = json.loads(sidecar.read_text())
    assert [p["frame_idx"] for p in report["panels"]] == [1, 0]
    assert report["panels"][0]["observed_joints"] == [17, 0]
    assert report["panels"][1]["observed_joints"] == [0, 0]
    assert report["panels"][0]["projectable_teacher_joints"] == [17, 0]
    assert not report["panels"][0]["projectable_teacher_ball"]
    assert not report["panels"][1]["observed_ball"]
