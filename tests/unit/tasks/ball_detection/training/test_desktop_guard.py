"""Independent D:/ext4 capacity bounds, RAM admission and emergency shutdown."""

import importlib.util
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
    monkeypatch.setattr(guard, "e_drive_available", lambda: 20 * GIB)
    monkeypatch.setattr(guard, "filesystem_used", lambda _: 200 * GIB)


@pytest.mark.parametrize("drive,path", [("d", "/mnt/d"), ("e", "/mnt/wsl/tennis-data")])
def test_statvfs_uses_d_drive_available_blocks(guard, monkeypatch, drive, path):
    paths = []

    def stat(path):
        paths.append(path)
        return SimpleNamespace(f_bavail=3, f_bfree=1000, f_frsize=4096)

    monkeypatch.setattr(guard.os, "statvfs", stat)
    monkeypatch.setattr(guard.os.path, "ismount", lambda target: target == Path(path))
    assert getattr(guard, drive + "_drive_available")() == 3 * 4096
    assert paths == [Path(path)]


def test_missing_data_mount_does_not_measure_the_root_volume(guard, monkeypatch):
    monkeypatch.setattr(guard.os.path, "ismount", lambda _: False)
    monkeypatch.setattr(guard.os, "statvfs", lambda _: pytest.fail("Must not measure an unmounted directory"))
    with pytest.raises(OSError, match="not mounted"):
        guard.e_drive_available()


@pytest.mark.parametrize("ram,disk,data,reason", [
    (2 * GIB, 10 * GIB, 10 * GIB, None),
    (2 * GIB, 10 * GIB - 1, 40 * GIB, "d_drive_available_space"),
    (2 * GIB, 40 * GIB, 10 * GIB - 1, "e_drive_available_space"),
    (2 * GIB - 1, 40 * GIB, 40 * GIB, "host_available_ram"),
])
def test_exact_lower_bounds_and_one_byte_below(guard, ram, disk, data, reason):
    assert guard.violation(ram, disk, data) == reason


@pytest.mark.parametrize("disk", [10 * GIB - 1, 10 * GIB])
@pytest.mark.parametrize("drive", ["d", "e"])
def test_preflight_refuses_low_d_space_before_cuda_calls(guard, monkeypatch, tmp_path, disk, drive):
    calls = []

    def device_count():
        calls.append("device")
        return 1

    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: guard.GPU)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction: calls.append(fraction))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=16 * GIB))
    monkeypatch.setenv("TENNIS_RUN_ID", "owned-job")
    monkeypatch.setenv("TRAINING_QUEUE_DIR", str(tmp_path / ".training_queue"))
    monkeypatch.setenv("TENNIS_SOURCE_COMMIT", "pushed")
    monkeypatch.setattr(guard.subprocess, "check_output", lambda args, **_: str(tmp_path / ".git") if args[-1] == "--git-common-dir" else "pushed")
    monkeypatch.setattr(guard, "available_ram", lambda: 5 * GIB)
    monkeypatch.setattr(guard, "d_drive_available", lambda: disk if drive == "d" else 20 * GIB)
    monkeypatch.setattr(guard, "e_drive_available", lambda: disk if drive == "e" else 20 * GIB)
    if disk < 10 * GIB:
        with pytest.raises(OSError, match=drive + "_drive_available_space"):
            guard.configure(tmp_path)
        assert calls == []
    else:
        result = guard.configure(tmp_path)
        assert result["d_drive_available_min_bytes"] == 10 * GIB
        assert result["e_drive_available_min_bytes"] == 10 * GIB
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


@pytest.mark.parametrize("failure", ["disk", "ram", "statvfs", "e_disk", "e_statvfs"])
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
    if failure.startswith("e_"):
        data_samples = 0

        def data():
            nonlocal data_samples
            data_samples += 1
            if data_samples > 1:
                if failure == "e_statvfs":
                    raise OSError("E: probe unavailable")
                return 10 * GIB - 1
            return 20 * GIB

        monkeypatch.setattr(guard, "e_drive_available", data)
    if failure == "ram":
        ram = iter([5 * GIB, 2 * GIB - 1])
        monkeypatch.setattr(guard, "available_ram", lambda: next(ram))

    def terminate(pid, sig):
        assert pid == os.getpid() and sig == signal.SIGTERM
        assert json.loads(path.read_text())["triggered"]
        terminated.set()

    monkeypatch.setattr(guard.os, "kill", terminate)
    with guard.host_guard(path, poll_interval=.001) as status:
        assert terminated.wait(2)
    assert status["trigger_reason"] == {"disk": "d_drive_available_space", "ram": "host_available_ram", "statvfs": "resource_measurement_failed", "e_disk": "e_drive_available_space", "e_statvfs": "resource_measurement_failed"}[failure]
    assert status["minimum_d_drive_available_bytes"] >= 0
    assert status["minimum_e_drive_available_bytes"] >= 0
    if failure == "statvfs":
        assert "D: probe unavailable" in status["error"]
    if failure == "e_statvfs":
        assert "E: probe unavailable" in status["error"]


def test_process_tree_includes_every_thread_child_and_grandchildren(guard, tmp_path):
    def process(pid, parent, pages, children):
        folder = tmp_path / str(pid)
        folder.mkdir()
        fields = ['S'] + ['0'] * 21
        fields[1], fields[21] = str(parent), str(pages)
        (folder / 'stat').write_text(f'{pid} (train (worker)) ' + ' '.join(fields))
        for tid, values in children.items():
            task = folder / 'task' / str(tid)
            task.mkdir(parents=True)
            (task / 'children').write_text(values)

    process(10, 1, 10, {10: '20 999', 11: '20 40'})
    process(20, 10, 20, {20: '30'})
    process(30, 20, 30, {30: ''})
    process(40, 99, 1000, {40: ''})
    metrics = guard.process_tree_memory(proc=tmp_path, root_pid=10, page_size=4096)
    assert metrics['process_tree_rss_bytes'] == 60 * 4096
    assert metrics['parent_rss_bytes'] == 10 * 4096
    assert metrics['children_rss_bytes'] == 50 * 4096
    assert metrics['process_tree_pids'] == [10, 20, 30]
    assert metrics['process_tree_process_count'] == 3


def test_missing_training_parent_does_not_report_zero_rss(guard, tmp_path):
    with pytest.raises(FileNotFoundError):
        guard.process_tree_memory(proc=tmp_path, root_pid=99, page_size=4096)


def resource_sample(ram, disk=20 * GIB, data=20 * GIB):
    return dict(host_available_bytes=ram, d_drive_available_bytes=disk, e_drive_available_bytes=data,
                process_tree_rss_bytes=3000, parent_rss_bytes=1000, children_rss_bytes=2000,
                process_tree_process_count=3, process_tree_pids=[10, 20, 30])


def test_six_gib_admits_without_sleep_or_wait_event(guard, tmp_path):
    def unexpected_sleep(_):
        pytest.fail('Exactly 6 GiB must admit the update')

    path = tmp_path / 'waits.jsonl'
    observer = guard.RamAdmission(path, sample=lambda: resource_sample(6 * GIB), sleep=unexpected_sleep)
    observer.before_update(1001)
    assert path.read_text() == ''
    assert observer.metrics()['ram_wait_seconds'] == 0
    assert observer.metrics()['process_tree_rss_bytes'] == 3000
    assert observer.metrics()['admitted_host_available_bytes'] == 6 * GIB


def test_wait_has_no_deadline_and_logs_one_thirty_minute_notice(guard, tmp_path, capsys):
    clock = SimpleNamespace(now=0.)
    readings = iter([6 * GIB - 1, 5 * GIB, 5 * GIB, 5 * GIB, 6 * GIB])

    def sleep(seconds):
        assert seconds == 1.
        clock.now += 1000.

    path = tmp_path / 'waits.jsonl'
    observer = guard.RamAdmission(path, sample=lambda: resource_sample(next(readings)),
                                  sleep=sleep, clock=lambda: clock.now, wall=lambda: 10000 + clock.now)
    observer.before_update(1001)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['event'] for row in rows] == ['ram_wait_started', 'ram_wait_over_30_minutes', 'ram_wait_resumed']
    assert all(row['update'] == 1001 and row['started_at'] == 10000 for row in rows)
    assert rows[0]['host_available_bytes'] == 6 * GIB - 1
    assert rows[-1]['host_available_bytes'] == 6 * GIB
    assert rows[-1]['elapsed_seconds'] == observer.last_wait_seconds == 4000.
    assert rows[-1]['process_tree_rss_bytes'] == 3000
    notices = [json.loads(line)['event'] for line in capsys.readouterr().out.splitlines()]
    assert notices.count('ram_wait_over_30_minutes') == 1


@pytest.mark.parametrize('failure', ['ram', 'disk', 'e_disk'])
def test_emergency_threshold_still_interrupts_an_admission_wait(guard, tmp_path, failure):
    clock = SimpleNamespace(now=0.)
    bad = {'ram': resource_sample(2 * GIB - 1), 'disk': resource_sample(5 * GIB, 10 * GIB - 1),
           'e_disk': resource_sample(5 * GIB, data=10 * GIB - 1)}[failure]
    samples = iter([resource_sample(5 * GIB), bad])

    def sleep(_):
        clock.now += 1

    path = tmp_path / 'waits.jsonl'
    observer = guard.RamAdmission(path, sample=lambda: next(samples), sleep=sleep,
                                  clock=lambda: clock.now, wall=lambda: clock.now)
    with pytest.raises(MemoryError, match={'ram': 'host_available_ram', 'disk': 'd_drive_available_space', 'e_disk': 'e_drive_available_space'}[failure]):
        observer.before_update(1001)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['event'] for row in rows] == ['ram_wait_started', 'ram_wait_interrupted']
    assert rows[-1]['elapsed_seconds'] == 1
