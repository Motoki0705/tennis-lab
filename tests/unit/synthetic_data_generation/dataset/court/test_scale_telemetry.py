from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.court.scale import runtime, telemetry


def test_l4_measurement_separates_sampled_peak_and_release(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "fixture")
    memory = iter([10, 50, 12, 10])

    def query(fields, *, kind="gpu"):
        return (
            []
            if kind == "compute-apps"
            else [["gpu-fixture", "NVIDIA L4", str(next(memory))]]
        )

    monkeypatch.setattr(telemetry, "_query", query)
    monitor = telemetry.NvidiaL4Telemetry(interval_seconds=30)
    monitor.begin()
    metrics = monitor.end()
    assert metrics["sampled_peak_bytes"] == 50 * 1024**2
    assert metrics["peak_is_lower_bound"] is True
    assert metrics["torch_peak_allocated_bytes"] is None
    released = monitor.released()
    assert (
        released["before"]["device_used_bytes"]
        == released["after"]["device_used_bytes"]
    )
    assert released["renderer_context_released"] is True
    assert released["compute_processes_after"] == []


def test_gpu_mismatch_or_live_process_cannot_be_reported_as_released(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "fixture")
    monkeypatch.setattr(
        telemetry, "_query", lambda *args, **kwargs: [["gpu", "NVIDIA T4", "0"]]
    )
    with pytest.raises(RuntimeError, match="NVIDIA L4"):
        telemetry.NvidiaL4Telemetry()
    live: list[list[str]] = []
    monkeypatch.setattr(
        telemetry,
        "_query",
        lambda *args, **kwargs: (
            live
            if kwargs.get("kind") == "compute-apps"
            else [["gpu", "NVIDIA L4", "0"]]
        ),
    )
    monitor = telemetry.NvidiaL4Telemetry()
    live.append(["42", "100"])
    with pytest.raises(RuntimeError, match="release verification failed"):
        monitor.released()
    with pytest.raises(RuntimeError, match="existing compute"):
        telemetry.NvidiaL4Telemetry()


def test_colab_entrypoint_cannot_bypass_the_local_queue(monkeypatch, tmp_path):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="training queue"):
        runtime.LocalNHTRenderer(tmp_path, fingerprint={})
    with pytest.raises(RuntimeError, match="Colab VM"):
        runtime.ColabNHTRenderer(tmp_path, fingerprint={})
    with pytest.raises(RuntimeError, match="Colab VM"):
        telemetry.NvidiaL4Telemetry()
    monkeypatch.setenv("COLAB_RELEASE_TAG", "fixture")
    pinned = {"schema": "fixture"}
    monkeypatch.setattr(runtime, "renderer_fingerprint", lambda: pinned)
    backend = runtime.ColabNHTRenderer(tmp_path, fingerprint=pinned)
    assert backend.batch_size == 1
    with pytest.raises(RuntimeError, match="training queue"):
        runtime.LocalNHTRenderer(tmp_path, fingerprint=pinned)
