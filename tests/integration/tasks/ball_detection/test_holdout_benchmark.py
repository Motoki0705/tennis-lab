"""Saved pose identity and independent CPU rescore of paired holdout arrays."""

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.evaluation.holdout_inference import FramePredictions
from src.tasks.ball_detection.evaluation.holdout_metrics import clip_references
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.benchmarks.ball_detection_holdout import pose_reference, sha256, summarize
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def fixture_clip(tmp_path):
    directory = tmp_path / "ball"
    write_store_clip(directory, "meiji/video_001/clip_000/cam0", [frame(i, ball()) for i in range(3)], source="meiji", split="test")
    store = BallFrameStore(directory)
    clip = replace(store.clips[0], camera_id="cam0", group_id="video_001")
    return clip, clip_references(store, clip)


def publish_pose(tmp_path, clip, media_hash="fixture"):
    root = tmp_path / "poses/video_001/clip_000"
    source = {"clip_id": "video_001/clip_000", "videos": [{
        "camera_id": "cam0", "sha256": media_hash, "width": clip.source_width,
        "height": clip.source_height, "num_frames": clip.frame_count, "fps": 30.0,
    }]}
    store = ClipStore(root, source)
    uv: NDArray[np.float32] = np.zeros((1, 3, 1, 17, 2), np.float32)
    uv[0, :, 0, 9] = (13, 24)
    uv[0, :, 0, 10] = (40, 40)
    confidence: NDArray[np.float32] = np.ones((1, 3, 1, 17), np.float32)
    confidence[0, 1] = 0.1
    pose = ObjectObservations(("cam0",), (64, 48), 30.0, uv, confidence,
                              np.asarray([[[True], [True], [False]]]), np.zeros((1, 1), np.int64))
    store.publish("pose_estimation/cam0", pose, ArtifactCodec(ObjectObservations), schema="person_poses",
                  version=1, identity={}, dependencies={}, provenance={})
    return root


def test_pose_alignment_checksum_and_missing_support(tmp_path):
    clip, ref = fixture_clip(tmp_path)
    poses = tmp_path / "poses"
    missing, record = pose_reference(ref, clip, poses, threshold=0.5)
    assert record["status"] == "missing_scene"
    assert np.isnan(missing.wrist_distance).all()
    root = publish_pose(tmp_path, clip)
    located, record = pose_reference(ref, clip, poses, threshold=0.5)
    np.testing.assert_allclose(located.wrist_distance, [5, np.nan, np.nan], equal_nan=True)
    assert record["known_distance_frames"] == 1
    index = json.loads((root / "scene.json").read_text())
    descriptor = root / index["artifacts"]["pose_estimation/cam0"]["path"]
    descriptor.write_text(descriptor.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        pose_reference(ref, clip, poses, threshold=0.5)


def test_pose_from_different_video_is_rejected(tmp_path):
    clip, ref = fixture_clip(tmp_path)
    publish_pose(tmp_path, clip, media_hash="different-video")
    with pytest.raises(ValueError, match="media/size/timeline"):
        pose_reference(ref, clip, tmp_path / "poses", threshold=0.5)


def test_saved_arrays_rescore_and_refuse_changed_frame_order(tmp_path):
    _, ref = fixture_clip(tmp_path)
    report = tmp_path / "report"
    report.mkdir()
    np.savez_compressed(report / "references.npz", **asdict(ref))
    protocol: dict[str, Any] = {"schema": "ball_meiji_holdout_v1", "frames": 3,
                "checkpoints": {"ft_e13": {}, "mixed_ft": {}},
                "references_sha256": sha256(report / "references.npz"),
                "metrics": {"score_threshold": 0.5, "distance_px": 20, "near_wrist_px": 100}}
    write_json_atomic(report / "protocol.json", protocol)
    prediction = FramePredictions(ref.uv.copy(), np.ones(3, np.float32), ref.uv[:, None].copy(),
                                  np.ones((3, 1), np.float32), np.ones((3, 1), bool),
                                  np.zeros(3, np.int64), np.arange(3, dtype=np.int64))
    manifest = {}
    for name in protocol["checkpoints"]:
        path = report / f"{name}.npz"
        np.savez_compressed(path, row=ref.row, **asdict(prediction))
        manifest[name] = {"file": path.name, "sha256": sha256(path)}
    write_json_atomic(report / "predictions.json", manifest)
    summarize(report)
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics["ft_e13"]["overall_observed"]["recall"] == 1
    assert metrics["ft_e13"] == metrics["mixed_ft"]
    assert (report / "comparison.md").is_file()
    np.savez_compressed(report / "mixed_ft.npz", row=ref.row[::-1], **asdict(prediction))
    with pytest.raises(ValueError, match="checksum"):
        summarize(report)
    manifest["mixed_ft"]["sha256"] = sha256(report / "mixed_ft.npz")
    write_json_atomic(report / "predictions.json", manifest)
    with pytest.raises(ValueError, match="identity/order"):
        summarize(report)
