"""Finite CPU-only buffered/direct read comparison; no retries or cache eviction.

O_DIRECT bypasses Linux page cache, not necessarily WSL host/device caches.
Agreement cannot identify a hardware/software root cause. No fork occurs while
direct I/O is in flight; its anonymous buffer uses MAP_SHARED, not MAP_PRIVATE.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib
import itertools
import json
import mmap
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAIN = Path("/home/kamimura/projects/tennis-lab")
LIVE = MAIN / "third_party/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
GOOD = (
    MAIN / "outputs/tennis_scene/analyze/meiji_read_stream_capture/s42-001/pass-001.bin"
)
BAD = (
    MAIN
    / "outputs/tennis_scene/analyze/meiji_v9_read_witness/s42-001/001-initial_pin.bin"
)
OUTPUT = MAIN / "outputs/tennis_scene/analyze/meiji_read_modes"
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "run-slcs-meiji-stream-byte-diff-v1/comparison.json"
)
OFFSET = 1896501248
LENGTH = 4096
INDEX = 417


class StatxTimestamp(ctypes.Structure):
    _fields_ = [
        ("sec", ctypes.c_int64),
        ("nsec", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
    ]


class Statx(ctypes.Structure):
    # Linux UAPI struct statx: 256 bytes; DIOALIGN fields at bytes 152/156.
    _fields_ = [
        ("mask", ctypes.c_uint32),
        ("blksize", ctypes.c_uint32),
        ("attributes", ctypes.c_uint64),
        ("nlink", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("spare0", ctypes.c_uint16),
        ("ino", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("blocks", ctypes.c_uint64),
        ("attributes_mask", ctypes.c_uint64),
        ("atime", StatxTimestamp),
        ("btime", StatxTimestamp),
        ("ctime", StatxTimestamp),
        ("mtime", StatxTimestamp),
        ("rdev_major", ctypes.c_uint32),
        ("rdev_minor", ctypes.c_uint32),
        ("dev_major", ctypes.c_uint32),
        ("dev_minor", ctypes.c_uint32),
        ("mnt_id", ctypes.c_uint64),
        ("dio_mem_align", ctypes.c_uint32),
        ("dio_offset_align", ctypes.c_uint32),
        ("spare3", ctypes.c_uint64 * 12),
    ]


def direct_alignment(fd: int) -> dict[str, int]:
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "statx"):
        raise RuntimeError("libc statx unavailable; direct diagnostic refused")
    if (
        ctypes.sizeof(Statx) != 256
        or Statx.dio_mem_align.offset != 152
        or Statx.dio_offset_align.offset != 156
    ):
        raise RuntimeError("Unsupported statx layout")
    function = libc.statx
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Statx),
    ]
    function.restype = ctypes.c_int
    result = Statx()
    # AT_EMPTY_PATH = 0x1000, STATX_DIOALIGN = 0x2000, using the open fd.
    if function(fd, b"", 0x1000, 0x2000, ctypes.byref(result)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if (
        not result.mask & 0x2000
        or not result.dio_mem_align
        or not result.dio_offset_align
    ):
        raise RuntimeError(
            "STATX_DIOALIGN unsupported or zero; direct diagnostic refused"
        )
    return {
        "statx_mask": result.mask,
        "memory_bytes": result.dio_mem_align,
        "offset_bytes": result.dio_offset_align,
    }


def now() -> str:
    return datetime.now(UTC).isoformat()


def fields(value: os.stat_result) -> dict[str, int]:
    return {
        key: int(getattr(value, f"st_{key}"))
        for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def hashes(data: bytes) -> dict[str, str]:
    return {
        "hashlib_sha256": hashlib.sha256(data).hexdigest(),
        "cpython_sha256": importlib.import_module("_sha256").sha256(data).hexdigest(),
    }


def atomic_ledger(output: Path, ledger: dict[str, Any]) -> None:
    temporary = output / "ledger.tmp"
    with temporary.open("x") as stream:
        json.dump(ledger, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "ledger.json")


def guard_output(output: Path, sources: list[Path], allowed: Path) -> Path:
    resolved = output.resolve()
    if (
        output.is_symlink()
        or output.exists()
        or not resolved.is_relative_to(allowed.resolve())
    ):
        raise ValueError(
            "Output must be a new directory inside the designated output root"
        )
    if any(
        source.resolve().is_relative_to(resolved)
        or resolved.is_relative_to(source.resolve())
        for source in sources
    ):
        raise ValueError("Output overlaps an input")
    return resolved


def read_block(
    path: Path,
    direct: bool,
    record: dict[str, Any],
    *,
    offset: int = OFFSET,
    length: int = LENGTH,
) -> bytes:
    record.update(
        path=str(path), started_at=now(), stat_before=fields(path.stat()), direct=direct
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if direct:
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("O_DIRECT unavailable; no fallback")
        flags |= os.O_DIRECT
    fd = os.open(path, flags)
    try:
        record["fd_before"] = fields(os.fstat(fd))
        record["accepted_flags"] = fcntl.fcntl(fd, fcntl.F_GETFL)
        if direct:
            if not record["accepted_flags"] & os.O_DIRECT:
                raise RuntimeError("O_DIRECT flag not accepted; no fallback")
            record["statx_dioalign"] = direct_alignment(fd)
            with mmap.mmap(
                -1, length, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            ) as buffer:
                address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
                record["alignment"] = {
                    "configured_bytes": LENGTH,
                    "address": address,
                    "offset": offset,
                    "length": length,
                    "map_shared": True,
                    "statx_dioalign_queried": True,
                }
                if address % LENGTH or offset % LENGTH or length % LENGTH:
                    raise RuntimeError("Direct I/O alignment failed")
                alignment = record["statx_dioalign"]
                if (
                    address % alignment["memory_bytes"]
                    or offset % alignment["offset_bytes"]
                    or length % alignment["offset_bytes"]
                ):
                    raise RuntimeError(
                        "STATX_DIOALIGN requirements not met; direct diagnostic refused"
                    )
                count = os.preadv(fd, [buffer], offset)
                data = buffer[:count]
            record["direct_supported"] = True
        else:
            data = os.pread(fd, length, offset)
        record["bytes_read"] = len(data)
        return data
    finally:
        record["fd_after"] = fields(os.fstat(fd))
        os.close(fd)
        record["path_after"] = fields(path.stat())
        record["finished_at"] = now()


def validate_saved(good: bytes, bad: bytes, index: int = INDEX) -> None:
    if len(good) != LENGTH or len(bad) != LENGTH:
        raise ValueError("Saved block length differs from evidence")
    differences = [i for i, (a, b) in enumerate(zip(good, bad, strict=True)) if a != b]
    if differences != [index] or good[index] != 0x9D or bad[index] != 0xBD:
        raise ValueError("Saved blocks do not reproduce the accepted one-bit evidence")


def run(
    output: Path,
    ledger: dict[str, Any],
    *,
    good: Path = GOOD,
    bad: Path = BAD,
    live: Path = LIVE,
    offset: int = OFFSET,
    index: int = INDEX,
) -> bool:
    blocks: dict[str, bytes] = {}
    ledger.update(
        status="running", reads=[], pairwise=[], started_at=now(), direct_supported=None
    )
    atomic_ledger(output, ledger)
    try:
        for name, path, direct in [
            ("saved_good", good, False),
            ("saved_bad", bad, False),
            ("live_buffered_before", live, False),
            ("live_direct", live, True),
            ("live_buffered_after", live, False),
        ]:
            record: dict[str, Any] = {"name": name}
            ledger["reads"].append(record)
            try:
                data = read_block(path, direct, record, offset=offset)
                snapshot = output / f"{name}.bin"
                with snapshot.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                record.update(snapshot=str(snapshot), **hashes(data))
                record["byte_at_offset"] = data[index] if index < len(data) else None
                if direct:
                    ledger["direct_supported"] = True
                if len(data) != LENGTH:
                    raise ValueError("Short read; no retry")
                if record["hashlib_sha256"] != record["cpython_sha256"]:
                    raise ValueError("Hash providers disagree")
                if (
                    not record["stat_before"]
                    == record["fd_before"]
                    == record["fd_after"]
                    == record["path_after"]
                ):
                    raise ValueError("Source identity or metadata changed")
                expected_stat = ledger.get("source_preflight_stats", {}).get(str(path))
                if expected_stat is not None and record["stat_before"] != expected_stat:
                    raise ValueError("Source changed since preflight")
                for previous in ledger["reads"][:-1]:
                    if (
                        previous["path"] == str(path)
                        and previous["stat_before"] != record["stat_before"]
                    ):
                        raise ValueError("Source changed between scheduled reads")
                blocks[name] = data
                if name == "saved_bad":
                    validate_saved(blocks["saved_good"], data, index)
                if name.startswith("live_"):
                    record["matches"] = (
                        "good"
                        if data == blocks["saved_good"]
                        else "bad"
                        if data == blocks["saved_bad"]
                        else "neither"
                    )
                ledger["pairwise"] = [
                    {
                        "left": a,
                        "right": b,
                        "equal": blocks[a] == blocks[b],
                        "differing_bytes": sum(
                            x != y for x, y in zip(blocks[a], blocks[b], strict=True)
                        ),
                    }
                    for a, b in itertools.combinations(blocks, 2)
                ]
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if direct:
                    ledger["direct_supported"] = record.get("direct_supported", False)
                raise
            finally:
                atomic_ledger(output, ledger)
        ledger["status"] = "completed"
        return True
    except Exception as exc:
        ledger.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return False
    finally:
        ledger["finished_at"] = now()
        atomic_ledger(output, ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Require CUDA_VISIBLE_DEVICES set to the empty string")
    output = guard_output(args.output, [LIVE, GOOD, BAD, EVIDENCE], OUTPUT)
    evidence_bytes = EVIDENCE.read_bytes()
    evidence = json.loads(evidence_bytes)["comparison"]
    if (
        not evidence["gates_passed"]
        or evidence["differing_bytes"] != 1
        or evidence["details"]
        != [
            dict(
                offset_zero_based=OFFSET + INDEX,
                left_byte=157,
                right_byte=189,
                xor=32,
                chunk_index=1808,
                chunk_offset=676257,
                page_size=4096,
                page_index=463013,
                page_offset=417,
            )
        ]
    ):
        raise ValueError("Unexpected source evidence")
    live_stat = fields(LIVE.stat())
    if any(
        live_stat[key] != value
        for key, value in {"dev": 2096, "ino": 71274, "size": 2549075546}.items()
    ):
        raise ValueError("Live source identity differs from accepted source")
    for path, source in zip((GOOD, BAD), evidence["sources"], strict=True):
        if str(path) != source["path"] or fields(path.stat()) != source["stat_after"]:
            raise ValueError("Saved source identity differs from comparison evidence")
    source_metadata = {}
    for name, expected_hash in json.loads(evidence_bytes)[
        "metadata_sha256_after"
    ].items():
        metadata_path = Path(name)
        allowed_metadata = (GOOD.with_name("audit.json"), BAD.with_name("ledger.json"))
        if metadata_path not in allowed_metadata:
            raise ValueError("Unexpected metadata input path")
        digest = hashes(metadata_path.read_bytes())
        if (
            digest["hashlib_sha256"] != expected_hash
            or digest["cpython_sha256"] != expected_hash
        ):
            raise ValueError("Source metadata differs from comparison evidence")
        source_metadata[name] = digest
    filesystem = subprocess.run(
        [
            "findmnt",
            "--json",
            "--target",
            str(LIVE),
            "--output",
            "TARGET,SOURCE,FSTYPE,OPTIONS",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    ledger = {
        "command": [sys.executable, *sys.argv],
        "git_commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence": {"path": str(EVIDENCE), **hashes(evidence_bytes)},
        "source_metadata": source_metadata,
        "kernel": os.uname().release,
        "filesystem_read_only_query": json.loads(filesystem.stdout),
        "live_preflight_stat": live_stat,
        "source_preflight_stats": {
            str(LIVE): live_stat,
            **{source["path"]: source["stat_after"] for source in evidence["sources"]},
        },
        "limitations": __doc__,
        "offset": OFFSET,
        "length": LENGTH,
        "difference_offset_zero_based": OFFSET + INDEX,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    output.mkdir(parents=True, exist_ok=False)
    return 0 if run(output, ledger) else 1


if __name__ == "__main__":
    raise SystemExit(main())
