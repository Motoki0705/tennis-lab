"""Published partial clips remain inspectable without model execution."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.visualize_component_store import Review
from src.tasks.plcs.model_io.person_association import PersonReIDResult
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.person_association import PlayerReIDOutput
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
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


def test_track_contact_sheet_samples_identity_handoffs() -> None:
    review = Review.__new__(Review)
    review.frame_count = 20
    boxes: np.ndarray = np.zeros((1, 20, 4), np.float32)
    boxes[..., 2] = 20
    boxes[..., 3] = 40
    boxes[:, 15:, 2] = 10
    observed: np.ndarray = np.ones((1, 20), bool)
    observed[:, 6:9] = False

    samples = review.track_samples(boxes, observed, [[2, 9]],
                                   [{"earlier_id": 2, "later_id": 9, "overlap_span_frames": 3}])

    assert {5, 7, 9, 11, 14, 15, 17}.issubset(samples)


def test_review_hides_descendants_of_superseded_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = {"clip_id": "stale-fixture", "videos": [
        {"camera_id": camera, "path": str(tmp_path / f"{camera}.mp4"),
         "num_frames": 4, "fps": 10., "width": 64, "height": 48} for camera in ("cam0", "cam1", "cam2")
    ]}
    store = ClipStore(tmp_path / "store", source)
    ball = BallDetectionOutput("cam0", np.arange(4, dtype=np.int64),
        np.zeros((4, 2), np.float32), np.zeros(4, np.float32),
        np.zeros(4, bool), np.zeros(4, np.uint8), "observed_acceptance_weight")
    old_ball = store.publish("ball_detection/cam0", ball, ArtifactCodec(BallDetectionOutput),
        schema="ball_detections", version=1, identity={"revision": 1}, dependencies={}, provenance={"origin": "test"})
    detection = PersonDetectionOutput("cam0", np.zeros(5, np.int64),
        np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    old_detection = store.publish("person_detection/cam0", detection, ArtifactCodec(PersonDetectionOutput),
        schema="person_detections", version=1, identity={"revision": 1},
        dependencies={"ball": old_ball}, provenance={"origin": "test"})
    tracking = PersonTrackingOutput("cam0", np.zeros(0, np.int64),
        np.zeros((0, 4, 4), np.float32), np.zeros((0, 4), bool), (), ())
    store.publish("person_tracking/cam0", tracking, ArtifactCodec(PersonTrackingOutput),
        schema="person_tracks", version=3, identity={"revision": 1},
        dependencies={"detections": old_detection}, provenance={"origin": "test"})
    store.publish("ball_detection/cam0", ball, ArtifactCodec(BallDetectionOutput),
        schema="ball_detections", version=1, identity={"revision": 2}, dependencies={}, provenance={"origin": "test"})
    monkeypatch.setattr(Review, "frame", lambda self, camera, index: np.zeros((48, 64, 3), np.uint8))

    report = Review(store.index_path, tmp_path / "review").build()

    manifest = json.loads((report.parent / "manifest.json").read_text())["components"]
    assert manifest["ball_detection/cam0"]["status"] == "rendered"
    assert manifest["person_detection/cam0"]["status"] == "stale"
    assert manifest["person_tracking/cam0"]["status"] == "stale"


def test_review_exports_raw_model_cosine_numbers(tmp_path: Path) -> None:
    cameras = ("cam0", "cam1", "cam2")
    source = {"clip_id": "cosine-fixture", "videos": [
        {"camera_id": camera, "path": str(tmp_path / f"{camera}.mp4"),
         "num_frames": 4, "fps": 10., "width": 64, "height": 48} for camera in cameras
    ]}
    store = ClipStore(tmp_path / "store", source)
    embeddings = torch.tensor([[[1., 0.]], [[.6, .8]], [[0., 0.]]])
    valid = torch.tensor([[True], [True], [False]])
    ids = torch.tensor([[0], [1], [-1]])
    local = torch.tensor([[1], [2], [-1]])
    value = PlayerReIDOutput(cameras, PersonReIDResult(ids, ids, local, embeddings, valid, .775))
    reference = store.publish("person_reid", value, ArtifactCodec(PlayerReIDOutput),
        schema="person_identities", version=1, identity={"model": 1}, dependencies={},
        provenance={"origin": "component"})

    report = Review(store.index_path, tmp_path / "review").build()

    raw = json.loads((report.parent / "reid_raw_cosine.json").read_text())
    assert raw["embedding_artifact_id"] == reference.artifact_id
    assert raw["cosine_threshold"] == .775
    assert raw["similarity"][0][1] == pytest.approx(.6)
    assert raw["similarity"][1][0] == pytest.approx(.6)
    assert "reid_raw_cosine.csv" in report.read_text()


def test_confirmed_review_labels_non_target_tracks(tmp_path: Path) -> None:
    cameras = ("cam0", "cam1", "cam2")
    source = {"clip_id": "target-fixture", "videos": [
        {"camera_id": camera, "path": str(tmp_path / f"{camera}.mp4"),
         "num_frames": 4, "fps": 10., "width": 64, "height": 48} for camera in cameras
    ]}
    store = ClipStore(tmp_path / "store", source)
    raw_ids = torch.tensor([[0, -1], [1, -1], [1, -1]])
    result = PersonReIDResult(raw_ids, raw_ids.clone(), torch.tensor([[1, 2]] * 3),
        torch.zeros((3, 2, 2)), torch.tensor([[True, True]] * 3), .775)
    store.publish("person_reid", PlayerReIDOutput(cameras, result), ArtifactCodec(PlayerReIDOutput),
        schema="person_identities", version=1, identity={"confirmed": True}, dependencies={},
        provenance={"origin": "confirmed_person_association"})

    review = Review(store.index_path, tmp_path / "review")

    assert review.confirmed_track_labels("cam0", np.array([1, 2])) == {1: 0, 2: -1}
