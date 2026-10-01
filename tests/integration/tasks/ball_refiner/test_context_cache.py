"""JPEG provenance, execution receipts and strict context-cache round trips."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.context_arrays import ContextArrays, GeneratedContext
from src.tasks.ball_refiner.data.context_cache import (
    ContextCache,
    generate_context_cache,
)
from src.tasks.ball_refiner.data.evidence_cache import (
    EvidenceCache,
    generate_evidence_cache,
)
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.refiner_2d import (
    Refiner2DConfig,
    Refiner2DInput,
    build_ball_refiner_2d,
    refiner_2d_nll,
)
from src.utils.checksum import dual_sha256
from tests.benchmarks.ball_refiner_context import verify_context
from tests.integration.tasks.ball_refiner.test_evidence_cache import (
    cache_inputs as cache_inputs,
)


class FakeProducer:
    def __init__(self):
        self.revision = 0

    def identity(self):
        return {"revision": self.revision, "test_models": True}

    def predict(self, store, clip):
        rows = store.clip_rows(clip)
        for row in rows:
            image = store.read_bgr(int(row))
            assert image.shape == (clip.height, clip.width, 3)
        n = clip.frame_count
        observed = np.ones((n, 2), bool)
        observed[2, 0] = False
        boxes = np.tile([15., 20., 16.], (n, 2, 1)).astype(np.float32)
        boxes[~observed] = 0
        keypoints = np.tile([-5., 15., 1.2], (n, 2, 17, 1)).astype(np.float32)
        keypoints[~observed] = 0
        arrays = ContextArrays(
            store.frames["frame_index"][rows].copy(), store.frames["pts"][rows].copy(),
            np.full(n, 2, np.int32), np.array([2, 7], np.int64), boxes, observed, keypoints,
            np.tile([10., 20.], (14, 1)).astype(np.float32), np.ones(14, bool),
        )
        return GeneratedContext(arrays, {
            "status": "complete", "person_region_policy": "full_frame",
            "person_detection_frames": n, "tracking_frames": n, "pose_crops": int(observed.sum()),
            "court_frame_indices": [0], "court_diagnostics": {"status": "ok"},
        })


@pytest.fixture
def evidence(cache_inputs):
    path = generate_evidence_cache(**cache_inputs)
    return EvidenceCache(path, BallFrameStore(cache_inputs["store_directory"]))


def test_context_roundtrip_preserves_raw_peaks_pts_and_source_pixel_mapping(evidence, tmp_path):
    producer = FakeProducer()
    directory = generate_context_cache(evidence, output=tmp_path / "context", producer=producer, clip_ids=None)
    cache = ContextCache(directory, evidence)
    cache.require_clips(evidence.clip_ids)
    clip = evidence.store.clips[0]
    restored = cache.load(clip.clip_id)
    expected = producer.predict(evidence.store, clip)
    for name, value in expected.arrays.arrays().items():
        np.testing.assert_array_equal(restored.arrays.arrays()[name], value)
    assert restored.execution == expected.execution
    assert cache.manifest["clips"][0]["jpeg_shard_sha256"] == evidence.manifest["clips"][0]["jpeg_shard_sha256"]
    assert cache.manifest["selection"]["scope"] == "all_evidence"
    # The original source is twice the stored dimensions; use inverse scale,
    # not endpoint normalization in stored pixels, and preserve outside poses.
    source = replace(clip, source_width=clip.width * 2, source_height=clip.height * 2)
    context = restored.arrays.model_context(source, pose_threshold=.15, provenance={"clip": clip.clip_id})
    pose, court = context.require_complete()
    np.testing.assert_allclose(pose.uv[0, 0, 0], [-10 / (source.source_width - 1), 30 / (source.source_height - 1)])
    np.testing.assert_allclose(court.uv[0], [20 / (source.source_width - 1), 40 / (source.source_height - 1)])
    assert not pose.valid[2, 0].any() and pose.valid[0].all()
    assert pose.confidence.max() == 1 and restored.arrays.keypoints[..., 2].max() > 1
    assert context.provenance["pose_confidence_saturated_slots"] == int(restored.arrays.track_observed.sum()) * 4
    with pytest.raises(FileExistsError):
        generate_context_cache(evidence, output=directory, producer=producer, clip_ids=None)
    with pytest.raises(ValueError, match="not generated"):
        cache.require_clips((*cache.clip_ids, "missing/clip"))
    # Feed the saved context and the paired detector cache to the real model.
    pose, court = restored.arrays.model_context(clip, pose_threshold=.15, provenance={}).require_complete()
    detector = evidence.load(clip.clip_id)
    batch = Refiner2DInput(
        detector.candidates, torch.from_numpy(detector.timestamps_seconds)[None],
        torch.from_numpy(pose.uv)[None], torch.from_numpy(pose.confidence)[None], torch.from_numpy(pose.valid)[None],
        torch.from_numpy(court.uv)[None], torch.from_numpy(court.confidence)[None], torch.from_numpy(court.valid)[None],
    )
    values = dict(OmegaConf.load(Path(__file__).resolve().parents[4] / "src/tasks/ball_refiner/configs/model/refiner_2d.yaml"))
    values.update(hidden_dim=16, attention_heads=2, dropout=0.0)
    pair = build_ball_refiner_2d(Refiner2DConfig(**values))
    loss = refiner_2d_nll(pair.run(batch), project_store_targets(evidence.store, clip).target(0, clip.frame_count)).loss
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in pair.model.parameters() if p.grad is not None)


@pytest.mark.parametrize("change", ["building", "rgb", "evidence", "store", "missing", "duplicate", "path", "shard", "pts", "receipt", "checksum", "keys", "nonfinite_peak", "filled_pose", "duplicate_id", "no_detection", "outside_court"])
def test_context_rejects_incomplete_corrupt_or_misaligned_data(evidence, tmp_path, change):
    directory = generate_context_cache(evidence, output=tmp_path / "context", producer=FakeProducer(), clip_ids=None)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["clips"][0]
    if change == "building":
        manifest["status"] = "building"
    elif change == "rgb":
        manifest["rgb_condition"] = "original_video"
    elif change == "evidence":
        manifest["evidence"]["manifest_sha256"] = "0" * 64
    elif change == "store":
        manifest["store"]["sha256"]["index.npz"] = "0" * 64
    elif change == "missing":
        manifest["clips"] = []
    elif change == "duplicate":
        manifest["clips"].append(record)
    elif change == "path":
        record["file"] = "../outside.npz"
    elif change == "shard":
        record["jpeg_shard_sha256"] = "0" * 64
    elif change == "receipt":
        record["execution"]["person_detection_frames"] = 0
    elif change == "checksum":
        record["sha256"] = "0" * 64
    else:
        path = directory / record["file"]
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if change == "pts":
            arrays["pts"] += 100
        elif change == "nonfinite_peak":
            arrays["keypoints"][0, 0, 0, 2] = np.nan
        elif change == "filled_pose":
            arrays["keypoints"][2, 0] = arrays["keypoints"][1, 0]
        elif change == "duplicate_id":
            arrays["track_ids"][:] = 0
        elif change == "no_detection":
            arrays["detection_count"][:] = 0
        elif change == "outside_court":
            arrays["court_points"][:] = -1
        else:
            arrays["unknown"] = np.zeros(1)
        np.savez_compressed(path, **arrays)
        record["sha256"] = dual_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ContextCache(directory, evidence).load(evidence.clip_ids[0])


@pytest.mark.parametrize("change", ["before_rgb", "during_rgb", "assets", "model_error"])
def test_generation_failure_leaves_no_consumable_context(evidence, tmp_path, change):
    shard = evidence.store.directory / "shards/clip-00000.bin"
    if change == "before_rgb":
        with shard.open("ab") as stream:
            stream.write(b"changed")

    class ChangingProducer(FakeProducer):
        def predict(self, store, clip):
            result = super().predict(store, clip)
            if change == "during_rgb":
                with shard.open("ab") as stream:
                    stream.write(b"changed")
            elif change == "assets":
                self.revision += 1
            elif change == "model_error":
                raise RuntimeError("model failed")
            return result

    directory = tmp_path / "context"
    with pytest.raises((ValueError, RuntimeError)):
        generate_context_cache(evidence, output=directory, producer=ChangingProducer(), clip_ids=None)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "building"
    with pytest.raises(ValueError, match="Incomplete"):
        ContextCache(directory, evidence)


def test_cli_rejects_relative_inputs_before_model_loading(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "src.tasks.ball_refiner.scripts.generate_context",
        "--store", "relative", "--evidence", str(tmp_path), "--output", str(tmp_path / "out"),
        "--scene-config", str(tmp_path / "scene.yaml"), "--max-tracks", "16", "--dry-run",
    ], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "All paths must be absolute" in result.stderr
    assert not (tmp_path / "out").exists()


def test_context_verifier_loads_in_a_separate_process(evidence, tmp_path):
    directory = generate_context_cache(evidence, output=tmp_path / "context", producer=FakeProducer(), clip_ids=None)
    report = tmp_path / "verification.json"
    root = Path(__file__).resolve().parents[4]
    environment = dict(os.environ, PYTHONPATH=str(root))
    command = [
        sys.executable, str(root / "tests/benchmarks/ball_refiner_context.py"),
        "--store", str(evidence.store.directory), "--evidence", str(evidence.directory),
        "--context", str(directory), "--report", str(report), "--pose-threshold", ".15",
    ]
    for clip_id in evidence.clip_ids:
        command.extend(["--clip-id", clip_id])
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(report.read_text())
    assert result["status"] == "verified_context_load"
    assert result["context_manifest_sha256"] == dual_sha256(directory / "manifest.json")
    assert len(result["clips"]) == len(evidence.clip_ids)
    assert all(row["tracks"] == 2 and row["court_valid_points"] == 14 for row in result["clips"])
    assert all(row["model_context_provenance"]["pose_confidence_saturated_slots"] > 0 for row in result["clips"])
    repeated = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr


def test_context_verifier_rejects_selection_and_changed_jpeg(evidence, tmp_path):
    directory = generate_context_cache(evidence, output=tmp_path / "context", producer=FakeProducer(), clip_ids=None)
    cache = ContextCache(directory, evidence)
    for selected in ((), (*cache.clip_ids, "other"), (*cache.clip_ids, cache.clip_ids[0])):
        with pytest.raises(ValueError, match="exactly the unique requested clips"):
            verify_context(cache, selected, pose_threshold=.15)
    with (evidence.store.directory / "shards/clip-00000.bin").open("ab") as stream:
        stream.write(b"changed after generation")
    with pytest.raises(ValueError, match="JPEG shard changed"):
        verify_context(cache, cache.clip_ids, pose_threshold=.15)
