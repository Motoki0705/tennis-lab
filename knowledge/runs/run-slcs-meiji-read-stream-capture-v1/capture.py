"""Finite CPU read-stream capture: tee the same immutable bytes into hashes/disk.

Each live chunk is read once, hashed (hashlib, _sha256, per-chunk hashlib), then
written to an exclusive snapshot and recorded. Snapshot verification uses only
saved bytes, never a recovery reread of live. Abnormal passes stop the schedule.
This narrows the gap of copying after failure; memory/CPU/I/O corruption can
still intervene between hashing, writing, and later verification. Agreement is
not a root-cause resolution. No numpy, torch, or model workload is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

PIN = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
CHUNK_SIZE = 1024 * 1024


class Digest(Protocol):
    def update(self, data: bytes) -> None: ...
    def hexdigest(self) -> str: ...


def independent() -> Digest:
    return cast(Digest, importlib.import_module("_sha256").sha256())


def stat_fields(value: os.stat_result) -> dict[str, int]:
    return {
        key: int(getattr(value, f"st_{key}"))
        for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def error_info(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "error": str(error)}


def read_stream(
    source: Path,
    snapshot: Path,
    expected: str = PIN,
    *,
    opener: Callable[[Path], BinaryIO] = lambda path: path.open("rb"),
    secondary_factory: Callable[[], Digest] = independent,
) -> dict[str, Any]:
    """One live open/read stream; injectable seams are for tiny CPU fixtures only."""
    report: dict[str, Any] = {
        "source": str(source),
        "snapshot": str(snapshot),
        "expected_sha256": expected,
        "chunks": [],
        "bytes_read": 0,
        "bytes_written": 0,
        "errors": [],
    }
    primary, secondary = hashlib.sha256(), secondary_factory()
    try:
        report["path_before"] = stat_fields(source.stat())
        with opener(source) as reader, snapshot.open("xb") as writer:
            report["descriptor_before"] = stat_fields(os.fstat(reader.fileno()))
            try:
                while True:
                    chunk = reader.read(CHUNK_SIZE)
                    if type(chunk) is not bytes:
                        raise TypeError(
                            "Reader must return immutable bytes; mutable buffers are rejected"
                        )
                    if not chunk:
                        break
                    offset = report["bytes_read"]
                    report["bytes_read"] += len(chunk)
                    primary.update(chunk)
                    secondary.update(chunk)
                    digest = hashlib.sha256(chunk).hexdigest()
                    written = writer.write(chunk)
                    report["bytes_written"] += written
                    report["chunks"].append(
                        {
                            "offset": offset,
                            "size": len(chunk),
                            "sha256": digest,
                            "written": written,
                        }
                    )
                    if written != len(chunk):
                        raise OSError("Short snapshot write")
            finally:
                writer.flush()
                os.fsync(writer.fileno())
                report["descriptor_after"] = stat_fields(os.fstat(reader.fileno()))
    except Exception as exc:
        report["errors"].append(error_info(exc))
    finally:
        try:
            report["path_after"] = stat_fields(source.stat())
        except OSError as exc:
            report["errors"].append(error_info(exc))
        report["hashlib_sha256"] = primary.hexdigest()
        report["cpython_sha256"] = secondary.hexdigest()
    checks = {
        "providers_agree": report["hashlib_sha256"] == report["cpython_sha256"],
        "expected_pin_matches": report["hashlib_sha256"]
        == expected
        == report["cpython_sha256"],
        "stat_unchanged": report.get("path_before") is not None
        and report.get("path_before")
        == report.get("descriptor_before")
        == report.get("descriptor_after")
        == report.get("path_after"),
        "byte_count_matches": report["bytes_read"]
        == report["bytes_written"]
        == report.get("path_before", {}).get("size"),
    }
    report["checks"] = checks
    report["status"] = (
        "passed" if all(checks.values()) and not report["errors"] else "failed"
    )
    return report


# Separate CPython process, isolated mode and no site packages; one saved-file read.
CHILD_CODE = """import hashlib, _sha256, json, sys
primary, independent = hashlib.sha256(), _sha256.sha256()
chunks, offset = [], 0
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        primary.update(chunk)
        independent.update(chunk)
        chunks.append({"offset": offset, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
        offset += len(chunk)
print(json.dumps({"hashlib_sha256": primary.hexdigest(), "cpython_sha256": independent.hexdigest(), "bytes_read": offset, "chunks": chunks}))
"""


def verify_snapshot(report: dict[str, Any]) -> None:
    """Exactly one child read plus one external sha256sum of saved bytes."""
    snapshot = Path(report["snapshot"])
    for name, command in (
        (
            "isolated_python",
            [sys.executable, "-I", "-S", "-c", CHILD_CODE, str(snapshot)],
        ),
        ("external_sha256sum", ["sha256sum", "--", str(snapshot)]),
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode:
                raise RuntimeError(f"{name} exit={result.returncode}: {result.stderr}")
            report[name] = (
                json.loads(result.stdout)
                if name == "isolated_python"
                else {"sha256": result.stdout.split()[0], "stdout": result.stdout}
            )
        except Exception as exc:
            report["errors"].append({"stage": name, **error_info(exc)})
    child, external = (
        report.get("isolated_python", {}),
        report.get("external_sha256sum", {}),
    )
    expected_chunks = [
        {key: value for key, value in chunk.items() if key != "written"}
        for chunk in report["chunks"]
    ]
    checks = {
        "saved_providers_match_capture": child.get("hashlib_sha256")
        == child.get("cpython_sha256")
        == external.get("sha256")
        == report["hashlib_sha256"]
        == report["cpython_sha256"],
        "saved_chunks_match_capture": child.get("chunks") == expected_chunks,
        "saved_length_matches_capture": child.get("bytes_read")
        == report["bytes_written"],
    }
    report["snapshot_checks"] = checks
    if not all(checks.values()) or report["errors"]:
        report["status"] = "failed"


def compare_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Read saved files once each; zero-based byte offsets, including length tails."""
    differences, offset, first = 0, 0, None
    with (
        Path(left["snapshot"]).open("rb") as a,
        Path(right["snapshot"]).open("rb") as b,
    ):
        while True:
            x, y = a.read(CHUNK_SIZE), b.read(CHUNK_SIZE)
            if not x and not y:
                break
            if x != y:
                for index in range(max(len(x), len(y))):
                    if index >= len(x) or index >= len(y) or x[index] != y[index]:
                        differences += 1
                        if first is None:
                            first = offset + index
            offset += max(len(x), len(y))
    left_chunks, right_chunks = left["chunks"], right["chunks"]
    changed_chunks = [
        index
        for index in range(max(len(left_chunks), len(right_chunks)))
        if index >= len(left_chunks)
        or index >= len(right_chunks)
        or left_chunks[index] != right_chunks[index]
    ]
    return {
        "left": left["snapshot"],
        "right": right["snapshot"],
        "equal": differences == 0,
        "different_bytes": differences,
        "first_difference_offset_zero_based": first,
        "different_chunk_indices": changed_chunks,
    }


def run_plan(
    checkpoint: Path, output: Path, passes: int, expected: str = PIN
) -> dict[str, Any]:
    """Execute only the finite schedule, stopping after its first anomaly."""
    if not 1 <= passes <= 3:
        raise ValueError("passes must be between 1 and 3")
    report: dict[str, Any] = {
        "status": "failed",
        "checkpoint": str(checkpoint),
        "expected_sha256": expected,
        "planned_passes": passes,
        "passes": [],
        "comparisons": [],
        "errors": [],
        "limitations": __doc__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "pid": os.getpid(),
    }
    try:
        for index in range(1, passes + 1):
            row = read_stream(checkpoint, output / f"pass-{index:03d}.bin", expected)
            report["passes"].append(row)
            # Retain and verify even partial/failed snapshots; never reopen live here.
            if Path(row["snapshot"]).exists():
                verify_snapshot(row)
            for previous in report["passes"][:-1]:
                comparison = compare_snapshots(previous, row)
                report["comparisons"].append(comparison)
                if not comparison["equal"] or comparison["different_chunk_indices"]:
                    row["status"] = "failed"
            if row["status"] != "passed":
                report["stop_reason"] = (
                    f"Pass {index} anomaly; remaining scheduled passes cancelled"
                )
                break
        report["status"] = (
            "passed"
            if len(report["passes"]) == passes
            and all(row["status"] == "passed" for row in report["passes"])
            else "failed"
        )
    except BaseException as exc:
        report["errors"].append(error_info(exc))
    finally:
        report["completed_passes"] = len(report["passes"])
        report["cancelled_passes"] = passes - len(report["passes"])
        with (output / "audit.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passes", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    # No CUDA or numerical libraries are used; reject accidental GPU exposure.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Run with CUDA_VISIBLE_DEVICES= for this CPU-only control")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = run_plan(
        args.checkpoint.absolute(), args.output_dir.absolute(), args.passes
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "planned_passes",
                    "completed_passes",
                    "cancelled_passes",
                )
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
