"""Real CPU checkpoint -> JPEG inference -> cache -> refiner NLL optimizer step."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import src.tasks.ball_refiner.data.evidence_cache as cache_module
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
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
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def cache_inputs(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = OmegaConf.create({
        "model": {"name": "conv_next_unet", "input_mode": "rgb", "in_channels": 3, "num_classes": 1,
                  "num_frames": 4, "input_layout": "bcthw", "dims": [4, 8, 16, 32], "depth": 1,
                  "drop_path_prob": 0.0, "mdd_a": 0.2, "mdd_b": 0.15},
        "data": {"image_size": [32, 32], "augmentation": {"normalize_imagenet": {
            "enabled": True, "mean": [0.1, 0.2, 0.3], "std": [0.5, 0.6, 0.7],
        }}},
    })
    pair = build_ball_detection_pair(cfg)
    checkpoint = tmp_path / "detector.ckpt"
    torch.save({"hyper_parameters": {"config": OmegaConf.to_container(cfg)},
                "state_dict": {f"model.{key}": value for key, value in pair.model.state_dict().items()}}, checkpoint)
    store = write_store_clip(tmp_path / "store", "tracknet/game1/clip1", [
        frame(i, ball("out_of_frame", None) if i == 1 else ball()) if i != 2 else frame(i)
        for i in range(9)
    ])
    arguments = dict(store_directory=store, checkpoint=checkpoint, output=tmp_path / "cache",
                     splits=("train",), sources=("tracknet",), device="cpu", subpixel_refine=True,
                     stride=2, batch_size=2, candidates=BallCandidateConfig())
    yield arguments
    torch.set_num_threads(previous)


def test_cache_roundtrip_and_refiner_optimizer_step(cache_inputs):
    directory = generate_evidence_cache(**cache_inputs)
    store = BallFrameStore(cache_inputs["store_directory"])
    cache = EvidenceCache(directory, store)
    clip = store.clips[0]
    evidence = cache.load(clip.clip_id)
    assert len(evidence.frame_index) == 9
    manifest = cache.manifest
    assert manifest["detector"]["image_normalization"] == {
        "enabled": True, "mean": [0.1, 0.2, 0.3], "std": [0.5, 0.6, 0.7],
    }
    assert manifest["detector"]["sha256"] == dual_sha256(cache_inputs["checkpoint"])
    assert manifest["clips"][0]["jpeg_shard_sha256"] == dual_sha256(store.directory / "shards/clip-00000.bin")
    assert manifest["context"] == {"pose": "not_generated", "court": "not_generated"}
    assert set(manifest["generator_sha256"]) == {"evidence.py", "evidence_inference.py", "evidence_cache.py"}
    config_path = Path(__file__).resolve().parents[4] / "src/tasks/ball_refiner/configs/model/refiner_2d.yaml"
    values = dict(OmegaConf.load(config_path))
    values.update(hidden_dim=16, attention_heads=2, dropout=0.0, use_pose=False, use_court=False)
    cfg = Refiner2DConfig(**values)
    pair = build_ball_refiner_2d(cfg)
    # Context is explicitly disabled. This does NOT convert absent context into full-model data.
    batch = Refiner2DInput(
        candidates=evidence.candidates, timestamps_seconds=torch.from_numpy(evidence.timestamps_seconds)[None],
        pose_uv=torch.zeros(1, 9, 0, 4, 2), pose_confidence=torch.zeros(1, 9, 0, 4),
        pose_valid=torch.zeros(1, 9, 0, 4, dtype=torch.bool), court_uv=torch.zeros(1, 14, 2),
        court_confidence=torch.zeros(1, 14), court_valid=torch.zeros(1, 14, dtype=torch.bool),
    )
    target = project_store_targets(store, clip).target(0, 9)
    original = {key: value.clone() for key, value in pair.model.state_dict().items()}
    optimizer = torch.optim.AdamW(pair.model.parameters(), lr=3e-4)
    loss = refiner_2d_nll(pair.run(batch), target).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in pair.model.parameters() if p.grad is not None)
    optimizer.step()
    assert any(not torch.equal(original[key], value) for key, value in pair.model.state_dict().items())
    before = (directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_evidence_cache(**cache_inputs)
    assert (directory / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("change", ["partial", "duplicate", "missing", "path", "store", "checksum", "seconds"])
def test_partial_misaligned_and_corrupt_cache_are_rejected(cache_inputs, change):
    directory = generate_evidence_cache(**cache_inputs)
    store = BallFrameStore(cache_inputs["store_directory"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["clips"][0]
    if change == "partial":
        manifest["status"] = "building"
    elif change == "duplicate":
        manifest["clips"].append(dict(record))
    elif change == "missing":
        manifest["clips"] = []
    elif change == "path":
        record["file"] = "../elsewhere.npz"
    elif change == "store":
        manifest["store"]["sha256"]["index.npz"] = "0" * 64
    elif change == "checksum":
        record["sha256"] = "0" * 64
    elif change == "seconds":
        path = directory / record["file"]
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["timestamps_seconds"] *= 2  # still increasing, but wrong time_base
        np.savez_compressed(path, **arrays)
        record["sha256"] = dual_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        EvidenceCache(directory, store).load(store.clips[0].clip_id)


def test_mutating_rgb_during_inference_leaves_unusable_partial_cache(cache_inputs, monkeypatch):
    original = cache_module.infer_clip_evidence

    def mutate(store, clip, predictor, **kwargs):
        result = original(store, clip, predictor, **kwargs)
        with (store.directory / "shards/clip-00000.bin").open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(cache_module, "infer_clip_evidence", mutate)
    with pytest.raises(ValueError, match="shard changed"):
        generate_evidence_cache(**cache_inputs)
    manifest = json.loads((cache_inputs["output"] / "manifest.json").read_text())
    assert manifest["status"] == "building"
    assert manifest["clips"] == []


def test_cli_uses_declared_paths_and_selection(cache_inputs):
    args = cache_inputs
    result = subprocess.run([
        sys.executable, "-m", "src.tasks.ball_refiner.scripts.generate_evidence",
        "--store", str(args["store_directory"]), "--checkpoint", str(args["checkpoint"]),
        "--output", str(args["output"]), "--sources", "tracknet", "--splits", "train",
        "--device", "cpu", "--stride", "2", "--batch-size", "2", "--max-candidates", "8",
        "--nms-kernel", "5", "--patch-size", "5", "--subpixel-refine",
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    cache = EvidenceCache(args["output"], BallFrameStore(args["store_directory"]))
    assert cache.clip_ids == ("tracknet/game1/clip1",)
    assert len(cache.load(cache.clip_ids[0]).frame_index) == 9
