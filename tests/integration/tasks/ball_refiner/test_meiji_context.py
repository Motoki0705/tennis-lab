"""Meiji-only input identities, whole-clip resume and explicit missing policy."""
import json

import numpy as np
import pytest

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context_cache import ContextCache
from src.tasks.ball_refiner.data.evidence_cache import (
    EvidenceCache,
    generate_evidence_cache,
)
from src.tasks.ball_refiner.data.meiji_context import (
    resume_meiji_cache,
    selection,
    write_meiji_plan,
)
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_refiner.test_context_cache import FakeProducer
from tests.integration.tasks.ball_refiner.test_evidence_cache import (
    cache_inputs as cache_inputs,
)
from tests.support.tasks.ball_detection.store import frame, write_store_clip


@pytest.fixture
def inputs(cache_inputs, tmp_path):
    for i in range(2):
        write_store_clip(cache_inputs["store_directory"], f"meiji/video_002/clip_00{i}/cam0",
                         [frame(f) for f in range(6)], source="meiji")
    path = cache_inputs["store_directory"] / "metadata.json"
    metadata = json.loads(path.read_text())
    for clip in metadata["clips"][1:]:
        clip["camera_id"] = "cam0"
    path.write_text(json.dumps(metadata))
    cache_inputs["sources"] = ("tracknet", "meiji")
    evidence = EvidenceCache(generate_evidence_cache(**cache_inputs), BallFrameStore(cache_inputs["store_directory"]))
    producer = FakeProducer()
    plan = write_meiji_plan(evidence, producer, tmp_path / "plan")
    return evidence, producer, plan


def test_resume_never_reruns_complete_clip_and_preserves_all17(inputs):
    evidence, producer, plan = inputs
    calls = []
    original = producer.predict

    def interrupted(store, clip):
        calls.append(clip.clip_id)
        if len(calls) == 2:
            raise RuntimeError("pose failed at source frame")
        return original(store, clip)

    producer.predict = interrupted
    digest = dual_sha256(plan)
    with pytest.raises(RuntimeError, match="pose failed"):
        resume_meiji_cache(evidence, producer, plan, digest)
    progress = json.loads((plan.parent / "progress.json").read_text())
    assert len(progress["completed"]) == 1
    assert progress["attempts"][1]["frame_range"] == [0, 6]
    assert progress["attempts"][1]["frame_status"] == "failed_not_published"
    directory = resume_meiji_cache(evidence, producer, plan, digest)
    assert len(calls) == 3 and calls[1] == calls[2]
    cache = ContextCache(directory, evidence)
    assert len(cache.clip_ids) == 2
    assert cache.manifest["coverage"]["absent"][0]["source"] == "tracknet"
    assert cache.manifest["coverage"]["absent"][0]["pose"] == "absent"
    with pytest.raises(ValueError, match="not generated"):
        cache.require_clips(evidence.clip_ids)
    for name in cache.clip_ids:
        restored = cache.load(name)
        assert restored.arrays.keypoints.shape == (6, 2, 17, 3)
        assert not restored.arrays.track_observed[2, 0]
    assert resume_meiji_cache(evidence, producer, plan, digest) == directory
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["model", "plan", "jpeg", "completed_npz", "completed_manifest"])
def test_resume_refuses_changed_identity_and_bytes(inputs, change):
    evidence, producer, plan = inputs
    digest = dual_sha256(plan)
    directory = resume_meiji_cache(evidence, producer, plan, digest)
    if change == "model":
        producer.revision += 1
    elif change == "plan":
        plan.write_text(plan.read_text() + " ")
    elif change == "jpeg":
        p = evidence.store.directory / "shards/clip-00001.bin"
        p.write_bytes(p.read_bytes() + b"changed")
    else:
        path = next((plan.parent / "attempts").glob("*/manifest.json"))
        if change == "completed_npz":
            path = path.parent / json.loads(path.read_text())["clips"][0]["file"]
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        resume_meiji_cache(evidence, producer, plan, digest)
    assert directory.exists()


def test_meiji_split_guard_blocks_video001_and_preserves_negative_raw_peaks(inputs):
    from dataclasses import replace

    evidence, producer, _ = inputs
    clip = evidence.store.clips[1]
    result = producer.predict(evidence.store, clip)
    raw = result.arrays.keypoints.copy()
    raw[0, 0, 7, 2] = -.2
    arrays = replace(result.arrays, keypoints=raw)
    model = arrays.model_context(clip, pose_threshold=.15, provenance={})
    assert arrays.keypoints[0, 0, 7, 2] == np.float32(-.2)
    assert not model.pose.valid[0, 0, 0]
    assert model.provenance["pose_confidence_negative_slots"] == 1
    evidence.store.clips = tuple(replace(c, clip_id=c.clip_id.replace("video_002", "video_001")) for c in evidence.store.clips)
    # Isolate metadata selection without opening any forbidden media.
    evidence.clip_ids = tuple(c.clip_id for c in evidence.store.clips)
    evidence.store.clip_by_id = lambda name: next(c for c in evidence.store.clips if c.clip_id == name)
    with pytest.raises(ValueError, match="identity"):
        selection(evidence)
