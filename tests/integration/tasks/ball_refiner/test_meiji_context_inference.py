"""Frozen shared sequence consumes real-row COCO17; no pose on GSI gaps."""
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import src.tasks.ball_refiner.data.meiji_context_inference as module
from src.submodules.models import PersonDetectionResult, Pose2DResult
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.court_kp import CourtImagePrediction
from tests.support.tasks.ball_detection.store import frame, write_store_clip


@pytest.fixture
def producer(tmp_path):
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "src/tennis_scene/configs"), version_base="1.3"):
        config = compose(config_name="pipeline", overrides=[f"paths.project_root={root}", "device=cpu",
                         "people_models.runtime.vitpose.batch_size=4"])
    scene = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    freeze = root / "knowledge/runs/run-i964-unseen-r16-20261001/freeze.json"
    return module.MeijiContextProducer(scene, freeze)


def test_refuses_nonfrozen_person_configuration(producer):
    with pytest.raises(ValueError, match="frozen"):
        module.MeijiContextProducer(replace(producer.scene, merge_duplicate_person_boxes=True), producer.freeze)


def test_verified_identity_cannot_ignore_changed_file_fingerprints(producer, monkeypatch):
    producer._verified_identity = {"assets": {"pose": "checked_hash"}}
    producer._verified_files = {"pose": (1, 2, 3, 4, 5)}
    current = dict(producer._verified_files)
    monkeypatch.setattr(producer, "_fingerprints", lambda: current)
    result = producer.identity()
    result["assets"]["pose"] = "caller_mutation"
    assert producer.identity()["assets"]["pose"] == "checked_hash"
    current["pose"] = (1, 2, 3, 4, 6)
    with pytest.raises(ValueError, match="changed"):
        producer.identity()


@pytest.mark.parametrize("fail", [False, True])
def test_shared_tracking_keeps_all_joints_and_missing_frames(producer, tmp_path, monkeypatch, fail):
    store_path = write_store_clip(tmp_path / "store", "meiji/video_002/clip_001/cam0",
                                 [frame(i) for i in range(9)], source="meiji")
    metadata = json.loads((store_path / "metadata.json").read_text())
    metadata["clips"][0]["camera_id"] = "cam0"
    (store_path / "metadata.json").write_text(json.dumps(metadata))
    store = BallFrameStore(store_path)
    calls: dict[str, Any] = {"detector": 0, "pose": 0, "sequence": 0, "unload": []}

    class Detector:
        def __init__(self, *args, **kwargs):
            assert kwargs["confidence"] == .3

        def predict(self, request):
            i = calls["detector"]
            calls["detector"] += 1
            boxes = np.array([[10, 5, 25, 40]], np.float32) if i != 4 else np.zeros((0, 4), np.float32)
            return PersonDetectionResult(boxes, np.ones(len(boxes), np.float32))

        def unload(self):
            calls["unload"].append("detector")

    class Pose:
        def __init__(self, *args, **kwargs):
            assert kwargs["batch_size"] == 4 and kwargs["precision"] == "float32"

        def predict(self, request):
            calls["pose"] += 1
            if fail:
                raise RuntimeError("pose runtime failure")
            points = torch.ones(len(request.frame_indices), 17, 3)
            points[..., 0] = torch.arange(17)
            points[..., 2] = 1.3
            points[:, 7, 2] = -.2
            return Pose2DResult(points)

        def unload(self):
            calls["unload"].append("pose")

    class Encoder:
        name = producer.scene.tracking.encoder
        input_size = (256, 128)

        def embed(self, crops):
            result = torch.zeros(len(crops), 1280)
            result[:, 0] = 1
            return result

    class AF:
        def __init__(self, path):
            pass

        def links(self, boxes, observed):
            return {i: i for i in range(len(boxes))}, []

    shared = module.track_sequence

    def sequence(*args, **kwargs):
        calls["sequence"] += 1
        assert kwargs["config"] == producer.scene.tracking
        return shared(*args, **kwargs)

    monkeypatch.setattr(module, "DinoPersonDetector", Detector)
    monkeypatch.setattr(module, "ViTPosePose2D", Pose)
    monkeypatch.setattr(module, "build_encoder", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(module, "AFLink", AF)
    monkeypatch.setattr(module, "track_sequence", sequence)
    monkeypatch.setattr(producer, "_court", lambda image: CourtImagePrediction(np.ones((14, 2), np.float32),
        np.ones(14, bool), {"status": "ok"}, None))
    monkeypatch.setattr(producer, "_select", lambda result, clip, court: (result.boxes, result.evidence, {"status": "complete"}))
    if fail:
        with pytest.raises(RuntimeError, match="pose runtime"):
            producer.predict(store, store.clips[0])
        assert producer.progress["stage"] == "pose_clip_tracking"
    else:
        result = producer.predict(store, store.clips[0])
        assert result.arrays.keypoints.shape[-2:] == (17, 3)
        assert result.arrays.track_observed.any()
        assert not result.arrays.track_observed[4].any()
        assert not result.arrays.keypoints[4].any()
        assert (result.arrays.keypoints[result.arrays.track_observed][:, 7, 2] < 0).all()
        assert result.execution["pose_frame_status"][4] == "no_detection"
        assert result.execution["inferred_detection_pose_crops"] == 8
        assert calls["pose"] == 8 and calls["sequence"] == 1
    assert calls["unload"] == ["detector", "pose"]
