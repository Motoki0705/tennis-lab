"""D: physical capacity, rather than WSL logical growth, bounds this campaign."""

import importlib.util
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

BUNDLE = Path("knowledge/runs/run-ball-arch-desktop-20261113")
GIB = 1024**3


@pytest.fixture
def guard(monkeypatch):
    def load(name):
        spec = importlib.util.spec_from_file_location(f"desktop_resource_{name}", BUNDLE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "rebuild_dataset", load("rebuild_dataset"))
    return load("guard")


def readings(guard, monkeypatch, tmp_path, disk):
    (tmp_path / "storage-baseline.json").write_text(json.dumps(dict(filesystem_used_bytes=0)))
    monkeypatch.setattr(guard, "available_ram", lambda: 5 * GIB)
    monkeypatch.setattr(guard, "d_drive_available", disk)
    monkeypatch.setattr(guard, "filesystem_used", lambda _: 200 * GIB)


def test_statvfs_uses_d_drive_available_blocks(guard, monkeypatch):
    paths = []

    def stat(path):
        paths.append(path)
        return SimpleNamespace(f_bavail=3, f_bfree=1000, f_frsize=4096)

    monkeypatch.setattr(guard.os, "statvfs", stat)
    assert guard.d_drive_available() == 3 * 4096
    assert paths == [Path("/mnt/d")]


@pytest.mark.parametrize("ram,disk,reason", [
    (4 * GIB, 10 * GIB, None),
    (4 * GIB, 10 * GIB - 1, "d_drive_available_space"),
    (4 * GIB - 1, 40 * GIB, "host_available_ram"),
])
def test_exact_lower_bounds_and_one_byte_below(guard, ram, disk, reason):
    assert guard.violation(ram, disk) == reason


@pytest.mark.parametrize("disk", [10 * GIB - 1, 10 * GIB])
def test_preflight_refuses_low_d_space_before_cuda_calls(guard, monkeypatch, tmp_path, disk):
    calls = []

    def device_count():
        calls.append("device")
        return 1

    cuda = SimpleNamespace(
        device_count=device_count,
        get_device_name=lambda: guard.GPU,
        set_per_process_memory_fraction=lambda fraction: calls.append(fraction),
        get_device_properties=lambda _: SimpleNamespace(total_memory=16 * GIB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("TENNIS_RUN_ID", "owned-job")
    monkeypatch.setenv("TRAINING_QUEUE_DIR", str(tmp_path / ".training_queue"))
    monkeypatch.setenv("TENNIS_SOURCE_COMMIT", "pushed")
    monkeypatch.setattr(guard.subprocess, "check_output", lambda args, **_: str(tmp_path / ".git") if args[-1] == "--git-common-dir" else "pushed")
    monkeypatch.setattr(guard, "available_ram", lambda: 5 * GIB)
    monkeypatch.setattr(guard, "d_drive_available", lambda: disk)
    if disk < 10 * GIB:
        with pytest.raises(OSError, match="d_drive_available_space"):
            guard.configure(tmp_path)
        assert calls == []
    else:
        result = guard.configure(tmp_path)
        assert result["d_drive_available_min_bytes"] == 10 * GIB
        assert calls == ["device", .8]


def test_observer_records_large_wsl_growth_without_stopping(guard, monkeypatch, tmp_path):
    observed = threading.Event()
    calls = []
    samples = 0

    def disk():
        nonlocal samples
        samples += 1
        if samples > 1:
            observed.set()
        return 10 * GIB

    readings(guard, monkeypatch, tmp_path, disk)
    monkeypatch.setattr(guard.os, "kill", lambda *_: calls.append("kill"))
    previous = signal.getsignal(signal.SIGTERM)
    with guard.host_guard(tmp_path / "host.json", poll_interval=.001) as status:
        assert observed.wait(2)
    assert not status["triggered"] and not calls
    assert status["maximum_filesystem_growth_bytes"] == 200 * GIB
    assert status["filesystem_growth_is_telemetry_only"]
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("failure", ["disk", "ram", "statvfs"])
def test_observer_persists_reason_then_terminates_on_runtime_failure(guard, monkeypatch, tmp_path, failure):
    terminated = threading.Event()
    samples = 0
    path = tmp_path / "host.json"

    def disk():
        nonlocal samples
        samples += 1
        if samples > 1 and failure == "statvfs":
            raise OSError("D: probe unavailable")
        return 10 * GIB - 1 if samples > 1 and failure == "disk" else 20 * GIB

    readings(guard, monkeypatch, tmp_path, disk)
    if failure == "ram":
        ram = iter([5 * GIB, 4 * GIB - 1])
        monkeypatch.setattr(guard, "available_ram", lambda: next(ram))

    def terminate(pid, sig):
        assert pid == os.getpid() and sig == signal.SIGTERM
        assert json.loads(path.read_text())["triggered"]
        terminated.set()

    monkeypatch.setattr(guard.os, "kill", terminate)
    with guard.host_guard(path, poll_interval=.001) as status:
        assert terminated.wait(2)
    assert status["trigger_reason"] == {"disk": "d_drive_available_space", "ram": "host_available_ram", "statvfs": "resource_measurement_failed"}[failure]
    assert status["minimum_d_drive_available_bytes"] >= 0
    if failure == "statvfs":
        assert "D: probe unavailable" in status["error"]
