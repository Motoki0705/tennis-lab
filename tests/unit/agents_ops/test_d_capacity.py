"""Physical D: measurement, watchdog, and fail-closed publication contracts."""

from __future__ import annotations

import base64
import importlib
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from xml.etree import ElementTree

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
GIB = 1 << 30
NOW_NS = 1_795_329_000_000_000_000
NOW_BOOTTIME_NS = 300 * 10**9
BOOT_ID = "f016a5a3-0a01-4104-a377-f7f933e7c885"
DRIVE_MOUNT = (
    r"137 85 0:72 / /mnt/d rw,noatime - 9p D:\134 rw,aname=drvfs;path=D:\;uid=1000"
)


@pytest.fixture
def capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    # Other agents-ops tests also expose the standalone lib directory.
    monkeypatch.syspath_prepend(str(OPS / "lib"))
    monkeypatch.syspath_prepend(str(OPS))
    mod = importlib.import_module("lib.d_capacity")
    monkeypatch.setattr(mod, "MOUNTINFO", tmp_path / "mountinfo")
    boot_id_file = tmp_path / "boot_id"
    boot_id_file.write_text(BOOT_ID + "\n")
    monkeypatch.setattr(mod, "BOOT_ID_FILE", boot_id_file)
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "default.toml")
    monkeypatch.delenv(mod.CONFIG_ENV, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(mod.time, "time_ns", lambda: NOW_NS)
    monkeypatch.setattr(mod.time, "clock_gettime_ns", lambda clock: NOW_BOOTTIME_NS)
    return mod


def test_d_capacity_reads_available_blocks_and_fragment_size(
    capacity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity.MOUNTINFO.write_text(DRIVE_MOUNT)

    def statvfs(path: str) -> SimpleNamespace:
        assert path == "/mnt/d"
        return SimpleNamespace(f_bavail=279, f_frsize=8192, f_bfree=400, f_bsize=4096)

    monkeypatch.setattr(os, "statvfs", statvfs)
    assert capacity.measure_available_bytes() == 2_285_568


@pytest.mark.parametrize(
    "mountinfo",
    [
        "85 1 8:1 / / rw - ext4 /dev/sda rw",
        DRIVE_MOUNT.replace(r"D:\134", r"E:\134"),
        DRIVE_MOUNT.replace("aname=drvfs;", "aname=shared;"),
        DRIVE_MOUNT.replace("/ /mnt/d", "/subdirectory /mnt/d"),
    ],
    ids=("missing", "another-drive", "non-drvfs-9p", "subdirectory"),
)
def test_d_capacity_rejects_unexpected_mount(
    mountinfo: str, capacity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity.MOUNTINFO.write_text(mountinfo)

    def statvfs(path: str) -> None:
        pytest.fail(f"attempted to measure unexpected mount {path}")

    monkeypatch.setattr(os, "statvfs", statvfs)
    with pytest.raises(capacity.CapacityError, match="not mounted as the Windows D:"):
        capacity.measure_available_bytes()


def test_d_capacity_refuses_a_statvfs_error_or_mount_replacement(
    capacity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity.MOUNTINFO.write_text(DRIVE_MOUNT)

    def statvfs_error(path: str) -> None:
        raise OSError("drive I/O failed")

    monkeypatch.setattr(os, "statvfs", statvfs_error)
    with pytest.raises(
        capacity.CapacityError, match="cannot measure.*drive I/O failed"
    ):
        capacity.measure_available_bytes()

    def replace_mount(path: str) -> SimpleNamespace:
        capacity.MOUNTINFO.write_text(DRIVE_MOUNT.replace("137 ", "138 "))
        return SimpleNamespace(f_bavail=1, f_frsize=4096)

    monkeypatch.setattr(os, "statvfs", replace_mount)
    with pytest.raises(capacity.CapacityError, match="changed during measurement"):
        capacity.measure_available_bytes()


def test_default_threshold_allows_equality_and_config_errors_refuse(
    capacity: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    available = 20 * GIB
    monkeypatch.setattr(capacity, "measure_available_bytes", lambda: available)
    capacity.require_capacity()
    available -= 1
    with pytest.raises(capacity.CapacityError, match="21474836479 bytes.*21474836480"):
        capacity.require_capacity()

    config = tmp_path / "custom.toml"
    monkeypatch.setenv(capacity.CONFIG_ENV, str(config))
    with pytest.raises(capacity.CapacityError, match="config .* is missing"):
        capacity.require_capacity()
    config.write_text("threshold_gib = 1.25\n")
    with pytest.raises(capacity.CapacityError, match="positive integer threshold_gib"):
        capacity.require_capacity()
    config.write_text("threshold_gib = 19\n")
    capacity.require_capacity()


def test_watchdog_publishes_errors_and_notifies_once_per_capacity_transition(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = importlib.import_module("d_capacity.watchdog")
    path = tmp_path / "status.json"
    toasts = []
    values = iter((20 * GIB, 19 * GIB, 18 * GIB, None, 20 * GIB, 22 * GIB))

    def measure() -> int:
        value = next(values)
        if value is None:
            raise capacity.CapacityError("missing mount")
        return value

    def notify(title: str, body: str) -> None:
        toasts.append((title, body))
        assert json.loads(path.read_text())["measured_at_ns"] == NOW_NS

    monkeypatch.setattr(capacity, "measure_available_bytes", measure)
    for expected, success in [
        ("allow", True),
        ("blocked", True),
        ("blocked", True),
        ("error", False),
        ("allow", True),
        ("allow", True),
    ]:
        assert watchdog.observe(path, notify) is success
        state = json.loads(path.read_text())
        assert state["decision"] == expected
        assert state["threshold_bytes"] == 20 * GIB
        if expected in {"blocked", "error"}:
            with pytest.raises(capacity.CapacityError, match="below the|missing mount"):
                capacity.require_capacity(path)
        else:
            capacity.require_capacity(path)
    assert [title for title, _ in toasts] == [
        "tennis-lab D: capacity low",
        "tennis-lab D: capacity recovered",
    ]
    assert path.stat().st_mode & 0o777 == 0o644


def test_watchdog_records_initial_low_even_if_toast_result_is_ambiguous(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    watchdog = importlib.import_module("d_capacity.watchdog")
    path = tmp_path / "status.json"
    monkeypatch.setattr(capacity, "measure_available_bytes", lambda: 0)
    notifications = []

    def notify(title: str, body: str) -> None:
        notifications.append(title)
        raise watchdog.ToastError("PowerShell timed out")

    assert watchdog.observe(path, notify) is False
    diagnostic = capsys.readouterr().err
    assert "PowerShell timed out" in diagnostic
    assert "delivery uncertain. Duplicate protection applied" in diagnostic
    path.unlink()
    assert watchdog.observe(path, notify) is True
    assert notifications == ["tennis-lab D: capacity low"]


def test_watchdog_retries_when_powershell_could_not_start(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    watchdog = importlib.import_module("d_capacity.watchdog")
    path = tmp_path / "status.json"
    monkeypatch.setattr(capacity, "measure_available_bytes", lambda: 0)
    attempts = []

    def popen(arguments: list[str], **kwargs: Any) -> MagicMock:
        attempts.append(arguments[0])
        if len(attempts) == 1:
            raise FileNotFoundError("WSL interop unavailable")
        proc = MagicMock(returncode=0)
        proc.communicate.return_value = ("", "")
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert watchdog.observe(path) is False
    assert "toast not sent. Will retry" in capsys.readouterr().err
    with pytest.raises(capacity.CapacityError, match="below the"):
        capacity.require_capacity(path)
    path.unlink()
    assert watchdog.observe(path) is True
    assert watchdog.observe(path) is True
    assert len(attempts) == 2


def test_watchdog_notifies_recovery_after_runtime_state_is_lost(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = importlib.import_module("d_capacity.watchdog")
    path = tmp_path / "status.json"
    available = 19 * GIB
    monkeypatch.setattr(capacity, "measure_available_bytes", lambda: available)
    titles: list[str] = []

    def notify(title: str, body: str) -> None:
        titles.append(title)

    assert watchdog.observe(path, notify)
    path.unlink()
    available = 20 * GIB
    assert watchdog.observe(path, notify)
    assert watchdog.observe(path, notify)
    assert titles == ["tennis-lab D: capacity low", "tennis-lab D: capacity recovered"]


def test_expired_status_stays_expired_after_wall_clock_rollback(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    elapsed = 0
    wall_elapsed = 0
    monkeypatch.setattr(capacity, "measure_available_bytes", lambda: 20 * GIB)
    monkeypatch.setattr(capacity.time, "time_ns", lambda: NOW_NS + wall_elapsed)
    monkeypatch.setattr(
        capacity.time, "clock_gettime_ns", lambda clock: 300 * 10**9 + elapsed
    )
    path.write_text(capacity.sample().to_json())
    capacity.require_capacity(path)
    elapsed = wall_elapsed = 91 * 10**9
    with pytest.raises(capacity.CapacityError, match="older than 90 seconds"):
        capacity.require_capacity(path)
    wall_elapsed = 10**9
    with pytest.raises(capacity.CapacityError, match="older than 90 seconds"):
        capacity.require_capacity(path)


def test_status_from_another_boot_is_refused(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    monkeypatch.setattr(capacity, "measure_available_bytes", lambda: 20 * GIB)
    path.write_text(capacity.sample().to_json())
    capacity.require_capacity(path)
    capacity.BOOT_ID_FILE.write_text("f016a5a3-0a01-4104-a377-f7f933e7c886\n")
    with pytest.raises(capacity.CapacityError, match="different boot"):
        capacity.require_capacity(path)


def test_ci_requires_fresh_coherent_watchdog_status_and_never_measures(
    capacity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    monkeypatch.setattr(
        capacity, "measure_available_bytes", lambda: pytest.fail("CI measured D:")
    )
    with pytest.raises(capacity.CapacityError, match="status is missing"):
        capacity.require_capacity(path)
    freshness_boundary = NOW_BOOTTIME_NS - 90 * 10**9
    data = {
        "schema_version": 2,
        "available_bytes": 20 * GIB,
        "threshold_bytes": 20 * GIB,
        "measured_at_ns": NOW_NS - 90 * 10**9,
        "boot_id": BOOT_ID,
        "boottime_ns": freshness_boundary,
        "decision": "allow",
        "error": None,
    }
    path.write_text(json.dumps(data))
    capacity.require_capacity(path)
    for update, message in [
        ({"boottime_ns": freshness_boundary - 1}, "older than 90 seconds"),
        ({"boottime_ns": NOW_BOOTTIME_NS + 1}, "future measurement time"),
        ({"available_bytes": 19 * GIB}, "invalid D: watchdog status"),
    ]:
        path.write_text(json.dumps({**data, **update}))
        with pytest.raises(capacity.CapacityError, match=message):
            capacity.require_capacity(path)
    path.write_text("{")
    with pytest.raises(capacity.CapacityError, match="cannot read D: watchdog status"):
        capacity.require_capacity(path)


@pytest.mark.parametrize(
    ("exit_code", "error_name"), [(9, "ToastError"), (10, "ToastNotSent")]
)
def test_windows_toast_uses_powershell_and_keeps_text_literal(
    capacity: Any, monkeypatch: pytest.MonkeyPatch, exit_code: int, error_name: str
) -> None:
    module = importlib.import_module("lib.windows_toast")
    title = "D: 空き容量 <threshold>"
    body = "A & B '$()'"

    def popen(arguments: list[str], **kwargs: Any) -> MagicMock:
        assert arguments[0].endswith("/powershell.exe")
        assert arguments[-2] == "-EncodedCommand"
        script = base64.b64decode(arguments[-1]).decode("utf-16-le")
        assert ".Show(" in script
        payload = re.search(r"FromBase64String\('([^']+)'\)", script)
        assert payload
        xml = ElementTree.fromstring(base64.b64decode(payload[1]))
        assert [element.text for element in xml.findall(".//text")] == [title, body]
        proc = MagicMock(returncode=exit_code)
        proc.communicate.return_value = ("", "notifier failed")
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(module.ToastError, match="notifier failed") as failure:
        module.send_windows_toast(title, body)
    assert type(failure.value).__name__ == error_name
