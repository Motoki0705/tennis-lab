"""Recorder contracts without models, video files or a GPU."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from src.utils.inference_recording import (
    InferenceRecorder,
    model_action,
    model_event,
    recording_context,
    span,
    temporary_directory,
    wall_breakdown,
)
from src.utils.video.prefetch import PrefetchIterator


def test_failure_keeps_model_pair_context_batch_and_valid_json(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"abc")
    output = tmp_path / "record.json"
    with pytest.raises(ValueError, match="prediction failed"), InferenceRecorder(output), recording_context(clip_id="video_000/clip_002", camera_id="cam3", node="pose/cam3"):
        attributes: dict[str, Any] = dict(weights={"checkpoint": weights}, device="cpu", dtype="float32", instance="model-1")
        with model_event("load", "pose", **attributes):
            pass
        try:
            with span("inference", batch_shape=[3, 3, 256, 192], batch_items=3):
                raise ValueError("prediction failed")
        finally:
            with model_event("unload", "pose", **attributes):
                pass
    record = json.loads(output.read_text())
    assert record["schema"] == "inference_recording_v1" and record["status"] == "failed"
    load, inference, unload = [e for e in record["events"] if e["stage"] != "measurement"]
    assert [e["stage"] for e in (load, inference, unload)] == ["model_load", "inference", "model_unload"]
    assert inference["status"] == "error" and inference["error_type"] == "ValueError"
    assert inference["attributes"]["batch_shape"] == [3, 3, 256, 192]
    assert inference["attributes"]["clip_id"] == "video_000/clip_002"
    assert inference["attributes"]["camera_id"] == "cam3"
    assert load["attributes"]["weights"]["checkpoint"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert load["attributes"]["instance"] == unload["attributes"]["instance"]
    assert record["models"][0]["load_count"] == record["models"][0]["unload_count"] == 1
    assert sum(record["stage_wall_seconds"].values()) == pytest.approx(record["total_wall_seconds"])
    assert all(0 <= e["start_ns"] <= e["end_ns"] for e in record["events"])
    assert all(0 <= e["thread_cpu_ns"] <= e["end_ns"] - e["start_ns"] for e in record["events"])
    assert all(e["process_cpu_ns"] >= 0 and e["native_thread_id"] == threading.get_native_id()
               for e in record["events"])
    assert record["timings"]["by_node"]["pose/cam3"]["wall_seconds"] > 0.


def test_wall_breakdown_partitions_nested_and_prefetched_work() -> None:
    # ns expressed as seconds for easy independent, hand-calculated expectations.
    events = [
        {"stage": "node", "start_ns": 0, "end_ns": 10_000_000_000, "depth": 0},
        {"stage": "postprocess", "start_ns": 1_000_000_000, "end_ns": 9_000_000_000, "depth": 1},
        {"stage": "decode", "start_ns": 2_000_000_000, "end_ns": 6_000_000_000, "depth": 2},
        {"stage": "inference", "start_ns": 4_000_000_000, "end_ns": 8_000_000_000, "depth": 2},
    ]
    assert wall_breakdown(events, 10_000_000_000) == {
        "unattributed": 2., "postprocess": 2., "decode": 2., "inference": 4.,
    }
    # A zero-length nested interval cannot consume its parent's remaining time.
    assert wall_breakdown([
        {"stage": "postprocess", "start_ns": 0, "end_ns": 10_000_000_000, "depth": 1},
        {"stage": "decode", "start_ns": 2_000_000_000, "end_ns": 2_000_000_000, "depth": 2},
    ], 10_000_000_000) == {"postprocess": 10., "unattributed": 0.}


def test_disabled_recording_never_reads_assets_or_creates_files(tmp_path: Path) -> None:
    class Model:
        @model_action("load", "fake", weights=lambda _: {"missing": tmp_path / "missing"}, device=lambda _: "cuda")
        def load(self) -> int:
            return 7

    assert Model().load() == 7
    with model_event("load", "fake", weights={"missing": tmp_path / "missing"}, device="cuda", dtype="float32", instance="1"), span("inference", batch_shape=[2, 3]):
        pass
    assert list(tmp_path.iterdir()) == []


def test_prefetch_carries_context_and_ends_spans_before_yield(tmp_path: Path) -> None:
    def frames():
        for i in range(3):
            with span("decode", frame_index=i):
                value = i + 5
            yield value

    with InferenceRecorder(tmp_path / "record.json") as recorder, recording_context(clip_id="clip-2"), span("node") as node:
        values = list(PrefetchIterator(frames(), max_prefetch=1))
        node["outcome"] = "executed"
    assert values == [5, 6, 7]
    decode = [e for e in recorder.events if e["stage"] == "decode"]
    node_event = next(e for e in recorder.events if e["stage"] == "node")
    assert len(decode) == 3
    assert all(e["parent_id"] == node_event["id"] and e["attributes"]["clip_id"] == "clip-2" for e in decode)
    assert all(e["thread_id"] != node_event["thread_id"] for e in decode)
    assert {e["thread_role"] for e in decode} == {"decode_prefetch"}
    waits = [e for e in recorder.events if e["stage"] == "decode_queue_wait"]
    assert {e["thread_role"] for e in waits if e["attributes"]["operation"] == "read"} == {"main"}
    assert {e["thread_role"] for e in waits if e["attributes"]["operation"] == "write"} == {"decode_prefetch"}
    samples = recorder.samples
    assert min(s["rss_bytes"] for s in samples) > 0
    assert samples[0]["process_cpu_percent"] is None and samples[-1]["process_cpu_percent"] >= 0.
    assert any(t["role"] == "main" for t in samples[-1]["threads"])


@pytest.mark.parametrize("synchronize", [True, False])
def test_nvml_sampling_is_independent_of_cuda_timing_synchronization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synchronize: bool,
) -> None:
    from types import SimpleNamespace

    from src.utils import inference_resources

    calls = []
    gpu = {"cuda:0": {"utilization_percent": 25, "used_bytes": 1000, "total_bytes": 2000}}
    nvml = SimpleNamespace(sample=lambda: gpu, close=lambda: calls.append("nvml_close"))
    monkeypatch.setattr(inference_resources, "NvmlSampler", lambda devices: nvml)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="test_device"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 20)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    output = tmp_path / "record.json"
    with InferenceRecorder(output, devices=("cuda:0",), synchronize_cuda=synchronize), span("inference", model="fake"):
        calls.append("forward")
    trace = json.loads(output.read_text())
    assert trace["resources"]["nvml"]["cuda:0"] == {"used_peak_bytes": 1000, "total_bytes": 2000}
    assert all(sample["gpu"] == gpu for sample in trace["resource_samples"])
    syncs = [e for e in trace["events"] if e["stage"] == "cuda_sync"]
    if synchronize:
        inference = next(e for e in trace["events"] if e["stage"] == "inference")
        assert [e["attributes"]["boundary"] for e in syncs] == ["before", "after"]
        assert all(e["parent_id"] == inference["id"] for e in syncs)
        assert calls == ["cuda:0", "forward", "cuda:0", "nvml_close"]
    else:
        assert syncs == [] and calls == ["forward", "nvml_close"]


def test_temporary_peak_includes_short_lived_files(tmp_path: Path) -> None:
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    with InferenceRecorder(tmp_path / "record.json", sample_interval=60) as recorder, temporary_directory(temporary):
        (temporary / "array").write_bytes(b"x" * 4097)
    (temporary / "array").unlink()
    assert recorder.resources["temporary_disk_peak_bytes"] == 4097
    assert recorder.resources["ram_rss_sampled_peak_bytes"] > 0


def test_changed_weights_fail_before_another_model_load(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"abc")
    with InferenceRecorder(tmp_path / "record.json") as recorder:
        with model_event("load", "fake", weights={"w": weights}, device="cpu", dtype="float32", instance="1"):
            pass
        weights.write_bytes(b"different bytes")
        with pytest.raises(ValueError, match="changed during recording"), model_event("load", "fake", weights={"w": weights}, device="cpu", dtype="float32", instance="2"):
            pytest.fail("Changed weights must not be loaded")
    assert sum(e["stage"] == "model_load" for e in recorder.events) == 1


def test_nested_recorder_rejected_and_context_restored_after_failure(tmp_path: Path) -> None:
    with InferenceRecorder(tmp_path / "first.json"), pytest.raises(RuntimeError, match="nested"), InferenceRecorder(tmp_path / "nested.json"):
        pass
    with InferenceRecorder(tmp_path / "second.json") as recorder, span("inference", batch_shape=[0, 3]):
        pass
    assert next(e for e in recorder.events if e["stage"] == "inference")["attributes"]["batch_items"] == 0
    assert not (tmp_path / "nested.json").exists()


@pytest.mark.parametrize("interval", [0., -1., float("nan"), float("inf"), -float("inf")])
def test_invalid_sampling_interval_fails_before_resources_start(tmp_path: Path, interval: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        InferenceRecorder(tmp_path / "record.json", sample_interval=interval)
    assert not list(tmp_path.iterdir())


def test_initialization_failure_restores_the_execution_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise OSError("resource source unavailable")

    broken = InferenceRecorder(tmp_path / "broken.json")
    monkeypatch.setattr(broken, "_sample", unavailable)
    with pytest.raises(OSError, match="resource source unavailable"), broken:
        pytest.fail("Initialization must fail before running inference")
    assert json.loads((tmp_path / "broken.json").read_text())["status"] == "failed"
    with InferenceRecorder(tmp_path / "next.json"):
        pass
    assert json.loads((tmp_path / "next.json").read_text())["status"] == "complete"


def test_final_sampling_failure_saves_error_record_and_restores_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise OSError("final resource sample unavailable")

    with pytest.raises(RuntimeError, match="resource sampling failed"), InferenceRecorder(tmp_path / "failed.json", sample_interval=60) as recorder:
        with span("decode"):
            pass
        monkeypatch.setattr(recorder, "_sample", unavailable)
    record = json.loads((tmp_path / "failed.json").read_text())
    assert record["status"] == "failed"
    assert "final resource sample unavailable" in record["measurement_errors"][0]
    assert any(e["stage"] == "decode" for e in record["events"])
    with InferenceRecorder(tmp_path / "next.json"):
        pass
