from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "read_modes_probe", Path(__file__).with_name("probe.py")
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def fixture_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    good = bytearray(4096)
    good[417] = 0x9D
    bad = bytearray(good)
    bad[417] = 0xBD
    paths = tuple(tmp_path / name for name in ("good", "bad", "live", "output"))
    paths[0].write_bytes(good)
    paths[1].write_bytes(bad)
    paths[2].write_bytes(good)
    paths[3].mkdir()
    return paths[0], paths[1], paths[2], paths[3]


def test_saved_corruption_stops_before_live(tmp_path: Path) -> None:
    good, bad, live, output = fixture_sources(tmp_path)
    data = bytearray(bad.read_bytes())
    data[5] = 1
    bad.write_bytes(data)
    ledger: dict[str, Any] = {}
    assert not probe.run(output, ledger, good=good, bad=bad, live=live, offset=0)
    assert len(ledger["reads"]) == 2
    assert "one-bit evidence" in ledger["error"]
    assert json.loads((output / "ledger.json").read_text())["status"] == "failed"


def test_short_read_saved_and_stops(tmp_path: Path) -> None:
    good, bad, live, output = fixture_sources(tmp_path)
    good.write_bytes(b"abc")
    ledger: dict[str, Any] = {}
    assert not probe.run(output, ledger, good=good, bad=bad, live=live, offset=0)
    assert len(ledger["reads"]) == 1
    assert (output / "saved_good.bin").read_bytes() == b"abc"
    assert "Short read" in ledger["error"]


def test_unsupported_direct_stops_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, bad, live, output = fixture_sources(tmp_path)
    original_open = probe.os.open
    direct_attempts = []

    def reject_direct(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & probe.os.O_DIRECT:
            direct_attempts.append(path)
            raise OSError(errno.EINVAL, "Unsupported direct I/O fixture")
        return int(original_open(path, flags, *args, **kwargs))

    monkeypatch.setattr(probe.os, "open", reject_direct)
    ledger: dict[str, Any] = {}
    assert not probe.run(output, ledger, good=good, bad=bad, live=live, offset=0)
    assert len(direct_attempts) == 1
    assert len(ledger["reads"]) == 4
    assert ledger["direct_supported"] is False
    assert not (output / "live_buffered_after.bin").exists()


def test_exact_finite_schedule_and_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, bad, live, output = fixture_sources(tmp_path)
    original = probe.read_block
    calls = []

    def fake_direct(
        path: Path, direct: bool, record: dict[str, Any], **kwargs: Any
    ) -> bytes:
        calls.append((path, direct))
        data: bytes = original(path, False, record, **kwargs)
        if direct:
            record["direct_supported"] = True
            return b"X" + data[1:]
        return data

    monkeypatch.setattr(probe, "read_block", fake_direct)
    ledger: dict[str, Any] = {}
    assert probe.run(output, ledger, good=good, bad=bad, live=live, offset=0)
    assert calls == [
        (good, False),
        (bad, False),
        (live, False),
        (live, True),
        (live, False),
    ]
    assert [row["matches"] for row in ledger["reads"][2:]] == [
        "good",
        "neither",
        "good",
    ]
    assert len(ledger["pairwise"]) == 10
    assert ledger["reads"][3]["hashlib_sha256"] == ledger["reads"][3]["cpython_sha256"]


def test_aligned_shared_direct_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, _, _, _ = fixture_sources(tmp_path)
    original_open = probe.os.open

    def strip_direct(path: Any, flags: int) -> int:
        return int(original_open(path, flags & ~probe.os.O_DIRECT))

    def fake_preadv(fd: int, buffers: list[Any], offset: int) -> int:
        assert offset == 0
        assert len(buffers) == 1 and len(buffers[0]) == 4096
        buffers[0][:] = probe.os.pread(fd, 4096, offset)
        return 4096

    monkeypatch.setattr(probe.os, "open", strip_direct)
    monkeypatch.setattr(probe.fcntl, "fcntl", lambda *args: probe.os.O_DIRECT)
    monkeypatch.setattr(probe.os, "preadv", fake_preadv)
    monkeypatch.setattr(
        probe,
        "direct_alignment",
        lambda fd: {"statx_mask": 0x2000, "memory_bytes": 512, "offset_bytes": 512},
    )
    record: dict[str, Any] = {}
    assert probe.read_block(good, True, record, offset=0) == good.read_bytes()
    assert record["alignment"]["address"] % 4096 == 0
    assert record["alignment"]["map_shared"] is True


def test_statx_unsupported_refuses_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, _, _, _ = fixture_sources(tmp_path)
    original_open = probe.os.open

    def strip_direct(path: Any, flags: int) -> int:
        return int(original_open(path, flags & ~probe.os.O_DIRECT))

    def unavailable(fd: int) -> dict[str, int]:
        raise RuntimeError("STATX_DIOALIGN unsupported")

    def forbidden(*args: Any) -> int:
        raise AssertionError("Must not read without DIOALIGN")

    monkeypatch.setattr(probe.os, "open", strip_direct)
    monkeypatch.setattr(probe.fcntl, "fcntl", lambda *args: probe.os.O_DIRECT)
    monkeypatch.setattr(probe, "direct_alignment", unavailable)
    monkeypatch.setattr(probe.os, "preadv", forbidden)
    with pytest.raises(RuntimeError, match="DIOALIGN unsupported"):
        probe.read_block(good, True, {}, offset=0)


def test_write_path_guards(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"untouched")
    with pytest.raises(ValueError):
        probe.guard_output(source, [source], tmp_path)
    with pytest.raises(ValueError):
        probe.guard_output(tmp_path.parent / "outside", [source], tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        probe.guard_output(link / "new", [source], tmp_path)
    assert probe.guard_output(tmp_path / "new", [source], tmp_path) == tmp_path / "new"
    assert source.read_bytes() == b"untouched"
