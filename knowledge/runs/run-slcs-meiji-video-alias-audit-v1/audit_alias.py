"""One-shot, pure-Python video alias/snapshot control; never retries a read."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.utils.checksum import dual_sha256  # noqa: E402

EXPECTED = "43fa76065af0efb29f9714101fab991edf46c3d5840a892b6a9caf767a75e636"
CHUNK = 1024 * 1024


def stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        key: int(getattr(value, f"st_{key}"))
        for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def stream_copy(source: Path, destination: Path) -> dict[str, Any]:
    before = stat(source)
    total = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        descriptor_before = os.fstat(reader.fileno())
        for chunk in iter(lambda: reader.read(CHUNK), b""):
            writer.write(chunk)
            total += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
        descriptor_after = os.fstat(reader.fileno())
    after = stat(source)
    return {
        "source": str(source),
        "snapshot": str(destination),
        "bytes_copied": total,
        "source_before": before,
        "source_after": after,
        "source_stat_unchanged": before == after,
        "descriptor_stat_unchanged": all(
            getattr(descriptor_before, f"st_{key}")
            == getattr(descriptor_after, f"st_{key}")
            for key in before
        ),
        "snapshot_stat": stat(destination),
    }


def compare_bytes(left: Path, right: Path) -> dict[str, Any]:
    different, offset, first = 0, 0, None
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            x, y = a.read(CHUNK), b.read(CHUNK)
            if not x and not y:
                break
            if x != y:
                for index in range(max(len(x), len(y))):
                    if index >= len(x) or index >= len(y) or x[index] != y[index]:
                        different += 1
                        if first is None:
                            first = offset + index
            offset += max(len(x), len(y))
    return {
        "equal": different == 0,
        "different_bytes": different,
        "first_difference_offset_zero_based": first,
        "compared_bytes": offset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--alias", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "failed",
        "expected_sha256": EXPECTED,
        "pid": os.getpid(),
        "python": sys.executable,
        "python_version": sys.version,
        "argv": sys.argv,
        "operations": {},
        "errors": [],
        "interpretation": "A separate-process snapshot control of current bytes. Agreement does not invalidate or resolve the prior failed feature audit; no root cause is inferred.",
    }

    def attempt(label: str, action: Callable[[], Any]) -> None:
        try:
            report["operations"][label] = action()
        except Exception as exc:
            report["errors"].append(
                {
                    "operation": label,
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "details": getattr(exc, "details", None),
                }
            )

    try:
        attempt("script_sha256_before", lambda: dual_sha256(Path(__file__)))
        attempt(
            "checksum_source_sha256",
            lambda: dual_sha256(ROOT / "src/utils/checksum.py"),
        )
        paths = {"original": args.original.absolute(), "alias": args.alias.absolute()}
        report["sources"] = {key: str(path) for key, path in paths.items()}
        for label, path in paths.items():
            attempt(f"{label}_stat_before", partial(stat, path))
        for label, path in paths.items():
            snapshot = output / f"{label}.mp4"
            attempt(
                f"{label}_copy",
                partial(stream_copy, path, snapshot),
            )
        for label, path in paths.items():
            snapshot = output / f"{label}.mp4"
            attempt(
                f"{label}_snapshot_dual_sha256",
                partial(dual_sha256, snapshot),
            )
            attempt(f"{label}_live_dual_sha256", partial(dual_sha256, path))

            def external(path: Path = path) -> dict[str, Any]:
                result = subprocess.run(
                    ["sha256sum", "--", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"sha256sum failed: {result.returncode}: {result.stderr}"
                    )
                return {
                    "sha256": result.stdout.split()[0],
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }

            attempt(f"{label}_external_sha256sum", external)
        attempt(
            "snapshot_byte_comparison",
            lambda: compare_bytes(output / "original.mp4", output / "alias.mp4"),
        )
        for label, path in paths.items():
            attempt(f"{label}_stat_after", partial(stat, path))
        attempt("script_sha256_after", lambda: dual_sha256(Path(__file__)))
        operations = report["operations"]
        matches = {
            key: value == EXPECTED
            for key, value in operations.items()
            if key.endswith("_dual_sha256")
        }
        matches.update(
            {
                key: value["sha256"] == EXPECTED
                for key, value in operations.items()
                if key.endswith("_external_sha256sum")
            }
        )
        report["digest_matches_expected"] = matches
        report["stat_unchanged"] = {
            label: operations.get(f"{label}_stat_before")
            == operations.get(f"{label}_stat_after")
            for label in paths
        }
        report["imported_nonstdlib_heavy_modules"] = [
            name
            for name in sys.modules
            if name.split(".")[0] in {"numpy", "torch"}
            or "dataset_pipeline.features" in name
        ]
        complete = (
            len(matches) == 6
            and all(matches.values())
            and operations.get("snapshot_byte_comparison", {}).get("equal") is True
            and all(report["stat_unchanged"].values())
            and operations.get("script_sha256_before")
            == operations.get("script_sha256_after")
            and not report["imported_nonstdlib_heavy_modules"]
        )
        for label in paths:
            copied = operations.get(f"{label}_copy", {})
            complete = (
                complete
                and copied.get("source_stat_unchanged") is True
                and copied.get("descriptor_stat_unchanged") is True
                and copied.get("bytes_copied")
                == copied.get("source_before", {}).get("size")
            )
        report["status"] = "passed" if complete and not report["errors"] else "failed"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        with (output / "audit.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "errors": report["errors"],
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
