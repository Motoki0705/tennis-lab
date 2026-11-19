"""CPU coverage for the benchmark supervisor and allocator metrics."""

from __future__ import annotations

import importlib
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    repo = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(
        str(repo / "knowledge/runs/run-i1757-video-batch-meiji")
    )
    return importlib.import_module("benchmark")


def supervise_cpu_worker(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    phase: Path,
    code: str,
    *,
    time_limit: float,
) -> dict[str, Any]:
    command = [sys.executable, "-B", "-c", code, str(phase)]
    start_process = subprocess.Popen

    def worker_only(args: list[str], **kwargs: Any) -> subprocess.Popen:
        assert args == command
        return start_process(args, **kwargs)

    launch = Mock(side_effect=worker_only)
    monkeypatch.setattr(subprocess, "Popen", launch)
    result: dict[str, Any] = benchmark.supervise_worker(
        phase,
        command,
        phase,
        heartbeat_timeout=30,
        time_limit=time_limit,
    )
    assert launch.call_count == 1
    assert result["monitor"] == "heartbeat"
    assert benchmark.read(phase / "health/worker.json") == result
    return result


def test_watchdog_stops_stale_heartbeat_and_saves_cpu_stack(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    code = """
import faulthandler
import json
import os
import signal
import sys
import time
from pathlib import Path

with Path(os.environ['TENNIS_STACK_FILE']).open('w') as stack:
    faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
    heartbeat = Path(sys.argv[1]) / 'worker' / 'heartbeat.json'
    heartbeat.parent.mkdir()
    temporary = heartbeat.with_suffix('.tmp')
    temporary.write_text(json.dumps({'monotonic': time.monotonic() - 60, 'operation': 'stalled model'}))
    temporary.replace(heartbeat)
    time.sleep(30)
"""
    result = supervise_cpu_worker(benchmark, monkeypatch, tmp_path, code, time_limit=4)
    assert result["status"] == "failed"
    assert result["error_kind"] == "worker_heartbeat_timeout"
    assert result["returncode"] == -signal.SIGTERM
    assert result["stack_obtained_before_sigterm"] is True
    anomaly = benchmark.read(tmp_path / "anomaly.json")
    assert anomaly["kind"] == "worker_heartbeat_timeout"
    assert anomaly["last_heartbeat"]["operation"] == "stalled model"


def test_watchdog_enforces_wall_limit_before_first_heartbeat(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = supervise_cpu_worker(
        benchmark, monkeypatch, tmp_path, "import time; time.sleep(30)", time_limit=0.1
    )
    assert result["status"] == "failed"
    assert result["returncode"] == -signal.SIGTERM
    assert result["error_kind"] == "time_limit"
    assert benchmark.read(tmp_path / "anomaly.json")["last_heartbeat"] is None


def test_watchdog_collects_clean_worker_exit(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = supervise_cpu_worker(
        benchmark, monkeypatch, tmp_path, "print('worker passed')", time_limit=4
    )
    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert (tmp_path / "worker.log").read_text() == "worker passed\n"
    assert not (tmp_path / "anomaly.json").exists()


@pytest.mark.parametrize("condition", ["serial", "reuse", "batched", "embedding_reuse"])
def test_probe_and_measurement_share_one_host_snapshot_and_watchdog(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    condition: str,
) -> None:
    monkeypatch.setenv("TENNIS_RUN_ID", "test-job")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    monkeypatch.setattr(benchmark, "verify_pins", Mock())
    monkeypatch.setattr(benchmark, "require_empty_store", Mock())
    monkeypatch.setattr(
        subprocess,
        "Popen",
        Mock(side_effect=AssertionError("Unexpected external process")),
    )
    gpu_memory = Mock(return_value=(14 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("torch.cuda.mem_get_info", gpu_memory)
    monkeypatch.setattr("torch.cuda.get_device_name", Mock(return_value="Test GPU"))
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3, available=24 * 1024**3),
    )
    monkeypatch.setattr(
        benchmark.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3)
    )

    def simulated_worker(phase: Path, *_: Any, **__: Any) -> dict[str, str]:
        if phase.name == "probe":
            assert gpu_memory.call_count == 0
            benchmark.write(
                phase / "host.json", benchmark.host_resources(tmp_path, "cuda:0")
            )
        return {"status": "complete", "monitor": "heartbeat"}

    monitor = Mock(side_effect=simulated_worker)
    monkeypatch.setattr(benchmark, "supervise_worker", monitor)
    for attempt in benchmark.attempts(tmp_path):
        benchmark.write(
            attempt / "measurement.json",
            {"outcomes": {clip: {"status": "generated"} for clip in benchmark.CLIPS}},
        )

    assert benchmark.run(tmp_path, condition, "probe", "test-head") == 0
    assert benchmark.run(tmp_path, condition, "measurement", "test-head") == 0

    gpu_memory.assert_called_once_with("cuda:0")
    probe, measurement = monitor.call_args_list
    assert [probe.args[0], measurement.args[0]] == [
        tmp_path / "probe",
        tmp_path / "measurement",
    ]
    assert probe.kwargs == {"heartbeat_timeout": 180, "time_limit": 480}
    assert measurement.kwargs == {"heartbeat_timeout": 180, "time_limit": 10800}
    host = benchmark.read(tmp_path / "probe/terminal.json")["host"]
    assert host["device_free_bytes"] == 14 * 1024**3
    assert benchmark.read(tmp_path / "measurement/terminal.json")["host"] == host


def test_measurement_rejects_a_probe_from_another_queue_job(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TENNIS_RUN_ID", "current-job")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    monkeypatch.setattr(benchmark, "verify_pins", Mock())
    monkeypatch.setattr(benchmark, "require_empty_store", Mock())
    monkeypatch.setattr(
        benchmark,
        "host_resources",
        Mock(side_effect=AssertionError("Unexpected GPU query")),
    )
    monkeypatch.setattr(
        benchmark,
        "supervise_worker",
        Mock(side_effect=AssertionError("Unexpected worker launch")),
    )
    benchmark.write(
        tmp_path / "probe/terminal.json",
        {
            "job_id": "old-job",
            "commit": "test-head",
            "condition": "serial",
            "status": "complete",
        },
    )

    with pytest.raises(RuntimeError, match="same queue job"):
        benchmark.run(tmp_path, "serial", "measurement", "test-head")


def test_summary_uses_allocator_peaks_and_dispatch_capacity(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = importlib.import_module("report")
    measured = benchmark.attempts(tmp_path)[0]
    peaks = {
        "cuda:0": {"allocated_peak_bytes": 200_000, "reserved_peak_bytes": 262_144}
    }
    benchmark.write(
        measured / "recording.json",
        {
            "total_wall_seconds": 5,
            "metadata": {"batching": {"enabled": True}},
            "events": [
                {
                    "stage": "inference",
                    "status": "ok",
                    "attributes": {
                        "model": "DINO",
                        "batch_shape": [1, 3, 800, 800],
                        "batch_items": 1,
                    },
                },
                {
                    "stage": "batch_dispatch",
                    "status": "ok",
                    "attributes": {
                        "model": "dino",
                        "batch_capacity": 2,
                        "batch_items": 1,
                    },
                },
            ],
            "status": "complete",
            "stage_wall_seconds": {},
            "models": [],
            "resources": {"cuda": peaks},
        },
    )
    records = dict.fromkeys(
        benchmark.CLIPS, SimpleNamespace(num_frames=6, num_cameras=4)
    )
    monkeypatch.setattr(
        report, "load_dataset_manifest", lambda _: SimpleNamespace(clips=records)
    )
    monkeypatch.setattr(report, "clip_evidence", lambda *_: {"scene": None})
    benchmark.write(
        measured / "measurement.json",
        {
            "recording": "recording.json",
            "wall_seconds": 6,
            "worker_startup_seconds": 2,
            "regime": "first",
            "outcomes": {},
        },
    )

    destination = tmp_path / "summary.json"
    assert report.summarize(tmp_path, destination)
    (measurement,) = benchmark.read(destination)["measurements"]
    assert measurement["resources"] == {"cuda": peaks}
    batches = {row["stage"]: row for row in measurement["batches"]}
    assert batches["unscoped"]["items_histogram"] == {"1": 1}
    assert batches["unscoped"]["fill_fraction"] is None
    assert batches["batch_dispatch"]["configured_cap"] == 2
    assert batches["batch_dispatch"]["fill_fraction"] == 0.5
