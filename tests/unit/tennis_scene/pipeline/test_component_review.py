"""Published partial clips remain inspectable without model execution."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.visualize_component_store import Review
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def test_partial_component_store_produces_review_with_provenance(
    tmp_path: Path,
) -> None:
    source = {"clip_id": "review-fixture", "videos": [
        {"camera_id": camera, "path": str(tmp_path / f"{camera}.mp4"),
         "num_frames": 4, "fps": 10., "width": 64, "height": 48} for camera in ("cam0", "cam1", "cam2")
    ]}
    writer = cv2.VideoWriter(str(tmp_path / "cam0.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 10., (64, 48))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((48, 64, 3), index * 20, np.uint8))
    writer.release()
    store = ClipStore(tmp_path / "clip" / "annotations" / "tennis_scene", source)
    ball = BallDetectionOutput("cam0", np.arange(4, dtype=np.int64),
        np.array([[20, 20], [21, 21], [22, 22], [0, 0]], np.float32),
        np.array([1, 0, 0, 0], np.float32),
        np.array([True, False, False, False]), np.array([1, 2, 3, 0], np.uint8),
        "observed_acceptance_weight")
    store.publish("ball_detection/cam0", ball, ArtifactCodec(BallDetectionOutput),
        schema="ball_detections", version=1, identity={"test": True}, dependencies={},
        provenance={"origin": "external_annotation"})
    report = Review(store.index_path, tmp_path / "review", videos=True).build()

    manifest = json.loads((report.parent / "manifest.json").read_text())
    assert manifest["components"]["ball_detection/cam0"]["status"] == "rendered"
    assert manifest["components"]["ball_detection/cam0"]["details"]["origin"] == "external_annotation"
    assert manifest["components"]["ball_detection/cam0"]["details"]["interpolated"] == "1"
    assert manifest["components"]["person_tracking/cam0"]["status"] == "missing"
    assert (report.parent / manifest["components"]["ball_detection/cam0"]["images"][0]).is_file()
    movie = report.parent / manifest["components"]["ball_detection/cam0"]["video"]
    assert movie.is_file()
    capture = cv2.VideoCapture(str(movie))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    capture.release()
