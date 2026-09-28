"""Context provenance and masks at the pipeline/refiner boundary."""

import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context import load_pipeline_context
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def clip(tmp_path):
    write_store_clip(tmp_path / "ball", "meiji/video_000/clip_000/cam0", [frame(i, ball()) for i in range(3)], source="meiji")
    return replace(BallFrameStore(tmp_path / "ball").clips[0], camera_id="cam0", group_id="video_000")


def publish(tmp_path, clip, *, court_frame=0, pose=True, court=True, media_hash="fixture", confidence=0.8):
    root = tmp_path / "pipeline"
    store = ClipStore(root, {"clip_id": "video_000/clip_000", "videos": [{
        "camera_id": "cam0", "sha256": media_hash, "width": 64, "height": 48,
        "num_frames": 3, "fps": 30.0,
    }]})
    if pose:
        uv: NDArray[np.float32] = np.zeros((1, 3, 1, 17, 2), np.float32)
        uv[0, :, 0, 7:11] = [[-10, 20], [30, 10], [40, 20], [50, 30]]
        conf: NDArray[np.float32] = np.full((1, 3, 1, 17), confidence, np.float32)
        conf[0, 2, 0, 10] = 0.1
        observation = ObjectObservations(("cam0",), (64, 48), 30.0, uv, conf,
                                         np.array([[[True], [False], [True]]]), np.array([[123]], np.int64))
        store.publish("pose_estimation/cam0", observation, ArtifactCodec(ObjectObservations),
                      schema="person_poses", version=1, identity={}, dependencies={}, provenance={})
    if court:
        observation = CourtKPResult(
            np.full((1, 1, 14, 2), 0.25, np.float32), np.ones((1, 1, 14), np.float32),
            np.array([court_frame], np.int32), {"temporal_policy": "static_first_frame", "source_frame_count": 3,
                                              "output_keypoint_contract": "camera_view_v2", "cameras": [{"camera_id": "cam0"}]},
        )
        store.publish("court_detection/cam0", observation, ArtifactCodec(CourtKPResult),
                      schema="court_observations", version=2, identity={}, dependencies={}, provenance={})
    return root / "scene.json"


def test_pose_order_observation_mask_and_static_court(clip, tmp_path):
    result = load_pipeline_context(clip, publish(tmp_path, clip), pose_threshold=0.5)
    pose, court = result.require_complete()
    assert pose.uv.shape == (3, 1, 4, 2)
    np.testing.assert_allclose(pose.uv[0, 0], np.array([[-10, 20], [30, 10], [40, 20], [50, 30]]) / [63, 47])
    assert pose.valid[0].all()  # outside-image elbow remains usable context
    assert not pose.valid[1].any()  # interpolated box is not a detected person
    assert pose.valid[2, 0].tolist() == [True, True, True, False]
    assert court.uv.shape == (14, 2)
    assert np.all(court.uv == 0.25)
    assert result.provenance["pose_frames"] == 2
    assert result.provenance["timeline"] == "exact_media_dense_frame_index_bound_to_store_pts"


def test_missing_execution_is_not_a_successful_all_missing_observation(clip, tmp_path):
    result = load_pipeline_context(clip, tmp_path / "absent/scene.json", pose_threshold=0.5)
    assert result.pose is None and result.provenance["pose_status"] == "missing_scene"
    with pytest.raises(ValueError, match="not generated"):
        result.require_complete()
    index = publish(tmp_path, clip, pose=False)
    result = load_pipeline_context(clip, index, pose_threshold=0.5)
    assert result.pose is None and result.court is not None
    assert result.provenance["pose_status"] == "not_generated"
    with pytest.raises(ValueError, match="not generated"):
        result.require_complete()


@pytest.mark.parametrize("field,value", [
    ("media_sha256", "other"), ("source_width", 128), ("frame_count", 4), ("fps", "60"),
])
def test_wrong_source_cannot_be_reused(clip, tmp_path, field, value):
    index = publish(tmp_path, clip)
    with pytest.raises(ValueError, match="media/size/timeline"):
        load_pipeline_context(replace(clip, **{field: value}), index, pose_threshold=0.5)


def test_nonzero_court_frame_is_rejected(clip, tmp_path):
    with pytest.raises(ValueError, match="frame-0"):
        load_pipeline_context(clip, publish(tmp_path / "court", clip, court_frame=1), pose_threshold=0.5)


def test_raw_heatmap_peaks_have_an_explicit_recorded_bounded_transform(clip, tmp_path):
    result = load_pipeline_context(clip, publish(tmp_path, clip, confidence=1.2), pose_threshold=0.5)
    assert result.pose is not None
    assert result.pose.confidence.max() == 1
    assert result.provenance["pose_confidence_saturated_slots"] == 11
    assert result.provenance["pose_confidence_raw_max"] == pytest.approx(1.2)


def test_array_corruption_is_not_downgraded_to_missing_context(clip, tmp_path):
    index = publish(tmp_path, clip)
    document = json.loads(index.read_text())
    descriptor = index.parent / document["artifacts"]["pose_estimation/cam0"]["path"]
    payload = json.loads(descriptor.read_text())
    array = descriptor.parent / next(iter(payload["arrays"]))
    with array.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_pipeline_context(clip, index, pose_threshold=0.5)
