from __future__ import annotations

import os
from pathlib import Path
from threading import Barrier, Lock

import pytest

from src.tasks.base.annotation.resources import measurements


def test_parallel_inventory_preserves_allocations_and_baseline_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    anchor = data / "anchor"
    anchor.write_bytes(b"a" * 8192)
    sparse = data / "sparse"
    with sparse.open("wb") as handle:
        handle.truncate(1024**2)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "not-owned").write_bytes(b"x" * 4096)
    (data / "link").symlink_to(outside, target_is_directory=True)
    roots = (str(data),)
    monkeypatch.setattr(measurements, "_parallel_metadata_roots", lambda _: ())
    baseline = tmp_path / "baseline.jsonl"
    measurements.write_baseline(baseline, roots)
    os.link(anchor, data / "survivor")
    anchor.unlink()
    (data / "new").write_bytes(b"n" * 4096)
    serial = measurements.storage_usage(roots, baseline, 123)
    expected = dict(measurements.allocations(roots))
    monkeypatch.setattr(measurements, "_parallel_metadata_roots", lambda value: value)
    actual = dict(measurements.allocations(roots))
    parallel = measurements.storage_usage(roots, baseline, 123)
    assert actual.keys() == expected.keys()
    assert str(outside / "not-owned") not in actual
    for path, stat in expected.items():
        observed = actual[path]
        assert (
            observed.st_dev,
            observed.st_ino,
            observed.st_blocks,
            observed.st_size,
        ) == (stat.st_dev, stat.st_ino, stat.st_blocks, stat.st_size)
    assert parallel.model_dump(exclude={"free_bytes"}) == serial.model_dump(
        exclude={"free_bytes"}
    )
    assert parallel.additional_bytes >= 8192 + 4096 + 123


def test_host_metadata_calls_overlap_without_omitting_entries(tmp_path: Path) -> None:
    for index in range(16):
        (tmp_path / str(index)).write_bytes(b"x")
    barrier, lock = Barrier(2), Lock()
    original = measurements._metadata_entry
    entered = 0

    def observed(path: str):
        nonlocal entered
        if Path(path).name.isdecimal():
            with lock:
                entered += 1
                first_pair = entered <= 2
            if first_pair:
                barrier.wait(timeout=5)
        return original(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(measurements, "_metadata_entry", observed)
        result = dict(measurements._parallel_allocations((str(tmp_path),)))
    assert entered == 16
    assert set(result) == {str(tmp_path)} | {
        str(tmp_path / str(index)) for index in range(16)
    }


def test_parallel_inventory_ignores_disappearance_but_rejects_permission_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "gone"
    gone.write_bytes(b"gone")
    denied = tmp_path / "denied"
    denied.write_bytes(b"denied")
    original = os.lstat

    def unavailable(path, *args, **kwargs):
        if str(path) == str(gone):
            raise FileNotFoundError(str(path))
        if str(path) == str(denied):
            raise PermissionError(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", unavailable)
    assert measurements._metadata_entry(str(gone)) is None
    with pytest.raises(PermissionError, match="denied"):
        list(measurements._parallel_allocations((str(tmp_path),)))


def test_parallel_inventory_counts_directory_that_disappears_during_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(_path):
        raise FileNotFoundError("directory removed after lstat")

    monkeypatch.setattr(os, "scandir", gone)
    result = dict(measurements._parallel_allocations((str(tmp_path),)))
    assert set(result) == {str(tmp_path)}


def test_parallel_mount_detection_uses_deepest_mount_and_decodes_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mountinfo = (
        "1 0 8:1 / / rw - ext4 /dev/root rw\n"
        "2 1 0:2 / /mnt/c rw - 9p C: rw\n"
        "3 2 8:1 / /mnt/c/local rw - ext4 /dev/root rw\n"
        "4 1 0:3 / /mnt/space\\040drive rw - 9p D: rw\n"
    )
    original = Path.read_text

    def read(path: Path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return mountinfo
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    roots = (
        "/mnt/c/source",
        "/mnt/c/local/data",
        "/mnt/c-other/source",
        "/home/data",
        "/mnt/space drive/data",
    )
    assert measurements._parallel_metadata_roots(roots) == (
        "/mnt/c/source",
        "/mnt/space drive/data",
    )
