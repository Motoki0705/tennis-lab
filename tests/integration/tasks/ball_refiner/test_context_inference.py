"""Generated context keeps every JPEG and distinguishes inference failure from absence."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

import src.tasks.ball_refiner.data.context_inference as module
from src.submodules.models import PersonDetectionResult, Pose2DResult
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.court_detection.inference.regions import CourtRegionUnavailable
from src.tennis_scene.pipeline.components.court_kp import CourtImagePrediction
from tests.support.tasks.ball_detection.store import frame, write_store_clip
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_court_kp_config,
    make_people_config,
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = BallFrameStore(write_store_clip(tmp_path / "store", "chat/source/clip", [frame(i) for i in range(6)], source="chat_annotation"))
    state: dict[str, Any] = {"detections": 0, "tracking": 0, "pose": 0, "court": 0, "unloaded": [], "mode": "normal"}

    class Detector:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"

        def predict(self, request):
            index = state["detections"]
            np.testing.assert_array_equal(request.frame_bgr, store.read_bgr(index))
            state["detections"] += 1
            count = 0 if index == 2 or state["mode"] == "empty" else 2
            boxes = np.array([[10., 8., 24., 40.], [30., 6., 48., 42.]], np.float32)[:count]
            return PersonDetectionResult(boxes, np.full(count, .9, np.float32))

        def unload(self):
            state["unloaded"].append("detector")

    class Tracker:
        def update(self, detection, image):
            state["tracking"] += 1
            return [{"id": 2 + i * 5, "bbx_xyxy": box} for i, box in enumerate(detection.boxes_xyxy)]

    class Pose:
        def __init__(self, *args, **kwargs):
            assert kwargs["precision"] == "float32"

        def predict(self, request):
            state["pose"] += 1
            assert request.frame_indices.tolist() == [0, 0, 1, 1, 3, 3, 4, 4, 5, 5]
            for index, image in request.frames_bgr:
                np.testing.assert_array_equal(image, store.read_bgr(index))
            keypoints = torch.zeros(len(request.frame_indices), 17, 3)
            keypoints[..., :2] = request.bbx_xys[:, None, :2]
            keypoints[..., 2] = 1.2
            if state["mode"] == "bad_pose":
                keypoints = keypoints[:-1]
            return Pose2DResult(keypoints)

        def unload(self):
            state["unloaded"].append("pose")

    class Court:
        def __init__(self, *args):
            pass

        def predict_image(self, rgb):
            state["court"] += 1
            np.testing.assert_array_equal(rgb, store.read_bgr(0)[..., ::-1])
            if state["mode"] == "court_absent":
                raise CourtRegionUnavailable(({"accepted": False, "status": "no_consensus"},))
            if state["mode"] == "court_error":
                raise RuntimeError("CUDA/model error")
            return CourtImagePrediction(np.full((14, 2), 10., np.float32), np.ones(14, bool), {"status": "ok"}, None)

        def unload(self):
            state["unloaded"].append("court")

    monkeypatch.setattr(module, "DinoPersonDetector", Detector)
    monkeypatch.setattr(module, "BotSortAssociator", Tracker)
    monkeypatch.setattr(module, "ViTPosePose2D", Pose)
    monkeypatch.setattr(module, "CourtKPModule", Court)
    producer = module.StoredJPEGContextProducer(people=make_people_config(tmp_path), court=make_court_kp_config(tmp_path), max_tracks=4)
    return store, producer, state


@pytest.mark.parametrize("mode", ["normal", "court_absent", "empty"])
def test_context_runs_full_timeline_and_only_observed_crops(setup, mode):
    store, producer, state = setup
    state["mode"] = mode
    result = producer.predict(store, store.clips[0])
    assert state["detections"] == state["tracking"] == 6
    assert state["court"] == 1 and state["unloaded"] == ["detector", "pose", "court"]
    assert store.clips[0].camera_id is None
    np.testing.assert_array_equal(result.arrays.pts, store.frames["pts"])
    if mode == "empty":
        assert state["pose"] == 0 and result.arrays.keypoints.shape == (6, 0, 17, 3)
        assert result.execution["pose_crops"] == 0
    else:
        assert state["pose"] == 1 and result.execution["pose_crops"] == 10
        assert not result.arrays.track_observed[2].any()
        assert not result.arrays.keypoints[2].any()
        assert result.arrays.track_ids.tolist() == [2, 7]
        assert result.arrays.keypoints[0, 1, 0, 0] > result.arrays.keypoints[0, 0, 0, 0]
    if mode == "court_absent":
        assert not result.arrays.court_valid.any()
        assert result.execution["court_diagnostics"]["status"] == "no_supported_region"
    else:
        assert result.arrays.court_valid.all()


@pytest.mark.parametrize("mode", ["bad_pose", "court_error", "track_capacity"])
def test_context_errors_are_not_converted_to_empty_observations(setup, mode):
    store, producer, state = setup
    state["mode"] = mode
    if mode == "track_capacity":
        producer.max_tracks = 1
    with pytest.raises((ValueError, RuntimeError)):
        producer.predict(store, store.clips[0])
    assert state["detections"] == 6 and "detector" in state["unloaded"]
    if mode == "bad_pose":
        assert "pose" in state["unloaded"] and state["court"] == 0
    if mode == "court_error":
        assert "court" in state["unloaded"]


def test_producer_requires_explicit_dino_and_matching_devices(setup):
    _, producer, _ = setup
    with pytest.raises(ValueError, match="DINO"):
        module.StoredJPEGContextProducer(people=replace(producer.people, detector="yolo"), court=producer.court, max_tracks=4)
    with pytest.raises(ValueError, match="common model device"):
        module.StoredJPEGContextProducer(people=producer.people, court=replace(producer.court, device="cuda"), max_tracks=4)


def test_context_identity_requires_extension_preflight_before_model_load(setup, monkeypatch):
    _, producer, state = setup

    def incompatible():
        raise RuntimeError("stale DINO extension")

    monkeypatch.setattr(module, "validate_dino_extension", incompatible)
    with pytest.raises(RuntimeError, match="stale DINO extension"):
        producer.identity()
    assert state["detections"] == state["pose"] == state["court"] == 0
