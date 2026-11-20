"""CPU checks for batch boundaries and the FP32 batch-one reference."""

import json
from types import MethodType, SimpleNamespace

import benchmark
import numpy as np
import pytest
import runtime
import torch
from batching import batch_slices, require_engine_profile
from clipreid import visual_forward_with_dynamic_batch

from src.tasks.player_association.appearance.encoders import _ClipVisual


def test_traced_clip_class_token_expands_to_the_runtime_batch():
    visual = _ClipVisual((2, 2), width=32, layers=1, heads=4, output_dim=16).eval()
    sample = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        expected = visual(sample[:2])
        visual.forward = MethodType(visual_forward_with_dynamic_batch, visual)
        traced = torch.jit.trace(visual, sample)
        actual = traced(sample[:2])
    assert actual[0].shape == (2, 32)
    torch.testing.assert_close(actual, expected)


def test_clip_tail_is_timed_separately_and_all_features_keep_input_order(monkeypatch):
    class Runner:
        def set_batch_size(self, size):
            self.pixels = torch.empty((size, 1, 1, 1))

        def run(self):
            return self.pixels.flatten(1).repeat(1, runtime.FEATURE_DIMENSION)

    runner = Runner()
    runner.set_batch_size(16)
    stages = []
    memory = SimpleNamespace(
        sample=stages.append, result=dict, driver="test", uuid="test"
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(runtime, "WARMUP", 1)
    monkeypatch.setattr(runtime, "BLOCKS", 1)
    monkeypatch.setattr(runtime, "REPEATS_PER_BLOCK", 2)
    data = np.arange(236, dtype=np.float32).reshape(-1, 1, 1, 1)
    groups = [data[part] for part in batch_slices(len(data), 16)]

    report, features = runtime.measure(runner, groups, memory)

    np.testing.assert_array_equal(features[:, 0], data.flatten())
    assert report["batch_shape"][0] == 16
    assert report["tail_batch"]["size"] == 12
    assert stages == [
        "after_warmup",
        "after_timed_block_1",
        "tail_after_warmup",
        "tail_after_timed_block_1",
        "after_accuracy",
    ]
    for result, size in ((report, 16), (report["tail_batch"], 12)):
        elapsed = result["latency_samples_ms"][0]
        assert result["images_per_second"] == pytest.approx(
            1000 * size * len(elapsed) / sum(elapsed)
        )


def test_clip_batched_fp32_compares_against_pinned_batch_one(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"weights")
    checkpoint_hash = benchmark.sha256_file(checkpoint)
    monkeypatch.setattr(benchmark, "CHECKPOINT_SHA256", checkpoint_hash)
    addresses = [
        benchmark.PROFILE_ADDRESS,
        ("video_000/clip_000", "cam0", 178, 1),
        ("video_000/clip_000", "cam1", 178, 0),
        ("video_000/clip_000", "cam1", 178, 1),
    ]
    tensors = {"eval": {"sha256": "fixed"}}
    monkeypatch.setattr(
        benchmark, "verify_inputs", lambda _: ({"tensors": tensors}, addresses)
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    features = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    np.save(reference / "features.npy", features)
    identity = {
        "model": benchmark.MODEL,
        "condition": "0",
        "batch_size": 1,
        "checkpoint_sha256": checkpoint_hash,
        "tensors": tensors,
        "profile_address": list(benchmark.PROFILE_ADDRESS),
    }
    receipt = reference / "result.json"
    reference_digest = benchmark.sha256_file(reference / "features.npy")
    benchmark.write_json(
        receipt, {"identity": identity, "features_sha256": reference_digest}
    )
    output = tmp_path / "batched"
    output.mkdir()

    def measure(command, log, timeout):
        assert command[command.index("--batch-size") + 1] == "2"
        benchmark.write_json(
            output / "measurement.json", {"batch_shape": [2, 3, 256, 128]}
        )
        np.save(output / "features.npy", features[:, ::-1])

    monkeypatch.setattr(benchmark, "run_stage", measure)
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-regression")
    args = SimpleNamespace(
        output=output,
        inputs=tmp_path,
        checkpoint=checkpoint,
        batch_size=2,
        adapter=None,
        model=benchmark.MODEL,
        condition="0",
        profile_from=None,
        measure_from=None,
        reference=reference,
        profiles="none",
        nsys="nsys",
        ncu="ncu",
    )

    result = benchmark.run_case(args)
    assert result["identity"]["profile_batch_addresses"] == [
        list(row) for row in addresses[:2]
    ]
    assert result["accuracy"]["cosine_p05"] == 0
    assert result["fp32_reference_features_sha256"] == reference_digest

    wrong_reference = json.loads(receipt.read_text())
    wrong_reference["identity"]["batch_size"] = 2
    benchmark.write_json(receipt, wrong_reference)
    with pytest.raises(ValueError, match="Reference identity mismatch"):
        benchmark.run_case(args)


def test_serialized_engine_must_have_the_requested_batch_profile():
    expected = {"pixels": [(1, 3, 256, 128), (16, 3, 256, 128), (32, 3, 256, 128)]}
    engine = SimpleNamespace(
        num_optimization_profiles=1,
        get_tensor_profile_shape=lambda name, index: expected[name],
    )
    assert require_engine_profile(engine, expected)["pixels"][-1][0] == 32
    engine.get_tensor_profile_shape = lambda name, index: (
        expected[name][:2] + [(16, 3, 256, 128)]
    )
    with pytest.raises(ValueError, match="differs from"):
        require_engine_profile(engine, expected)
