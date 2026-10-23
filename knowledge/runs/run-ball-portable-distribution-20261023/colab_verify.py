"""Download a fresh distribution, resume a deliberate interruption, then verify."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.distribution import (
    load_distribution,
    sample_windows,
    verify_distribution,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def download(remote: str, destination: Path, receipts: Path, *, attempt: str, seconds: int, resume: bool) -> dict[str, Any]:
    argv = [sys.executable, ".agents/skills/tennis-drive/scripts/drive.py", "--timeout-seconds", str(seconds),
            "download", remote, str(destination), "--verify"]
    if resume:
        argv.append("--overwrite")
    started_at, started = datetime.now(UTC).isoformat(), time.monotonic()
    with (receipts / f"{attempt}.stdout.json").open("wb") as out, (receipts / f"{attempt}.stderr.log").open("wb") as err:
        result = subprocess.run(argv, stdout=out, stderr=err, check=False)
    receipt = dict(schema_version=1, argv=argv, started_at=started_at,
                   elapsed_seconds=time.monotonic() - started, returncode=result.returncode)
    write(receipts / f"{attempt}.timing.json", receipt)
    return receipt


def reconcile_partial(destination: Path, partial: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove only recorded rclone temporaries after their final target verifies."""
    files = {e["path"]: e for e in manifest["files"]}
    files["distribution.json"] = dict(size_bytes=(destination / "distribution.json").stat().st_size,
                                      sha256=dual_sha256(destination / "distribution.json"))
    retained = restarted = 0
    removed = []
    for entry in partial:
        name = entry["path"]
        path = destination / name
        if name in files:
            if entry["size_bytes"] == files[name]["size_bytes"] and entry["sha256"] == files[name]["sha256"]:
                retained += 1
            else:
                restarted += 1
        elif path.is_file():
            candidates = [n for n in files if Path(n).parent == Path(name).parent
                          and Path(name).name.startswith(Path(n).name + ".") and name.endswith(".partial")]
            if (len(candidates) != 1 or dual_sha256(path) != entry["sha256"]
                    or dual_sha256(destination / candidates[0]) != files[candidates[0]]["sha256"]):
                raise ValueError("Unexpected or changed partial-copy file; refuse automatic cleanup")
            path.unlink()
            removed.append(entry)
    return dict(retained_complete_files=retained, restarted_incomplete_files=restarted,
                removed_known_incomplete_temporaries=removed)


def run(args: argparse.Namespace) -> None:
    if not os.environ.get("RCLONE_CONFIG"):
        raise ValueError("The Colab worker must supply its managed Drive configuration")
    if args.receipts.exists() or args.destination.exists() != args.resume_existing:
        raise FileExistsError("Use a new receipt directory; explicitly select fresh retrieval or resume-existing")
    args.receipts.mkdir(parents=True)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write(args.receipts / "environment.json", dict(
        schema_version=1, observed_at=datetime.now(UTC).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=sys.version, disk_before_download=shutil.disk_usage(args.destination.parent)._asdict(),
        uv_lock_sha256=dual_sha256(Path("uv.lock")),
    ))
    manifest_fetch = download(args.drive_path + "/distribution.json", args.receipts / "distribution.json",
                              args.receipts, attempt="manifest-download", seconds=1800, resume=False)
    if manifest_fetch["returncode"] or dual_sha256(args.receipts / "distribution.json") != args.manifest_sha256:
        raise ValueError("Drive manifest differs from the pinned local manifest")
    manifest = load_distribution(args.receipts)
    total = sum(e["size_bytes"] for e in manifest["files"]) + (args.receipts / "distribution.json").stat().st_size
    if total != args.expected_size_bytes:
        raise ValueError("Expected distribution size differs from its pinned manifest")
    existing = sum(e["size_bytes"] for e in manifest["files"]
                   if (args.destination / e["path"]).is_file()
                   and (args.destination / e["path"]).stat().st_size == e["size_bytes"])
    required = total - existing + max(e["size_bytes"] for e in manifest["files"])
    if shutil.disk_usage(args.destination.parent).free < required:
        raise ValueError("VM has insufficient free space for the fixed distribution and transfer temporary")
    # The Drive command owns rclone's timeout. A partial local directory is
    # inventoried and checked before --overwrite is used to resume that copy.
    interrupted = (download(args.drive_path, args.destination, args.receipts,
                            attempt="download-interrupted", seconds=45, resume=False)
                   if args.interrupt_download else None)
    partial = [dict(path=p.relative_to(args.destination).as_posix(), size_bytes=p.stat().st_size,
                    sha256=dual_sha256(p)) for p in args.destination.rglob("*") if p.is_file()]
    write(args.receipts / "download-partial.json", dict(
        schema_version=1, files=partial, returncode=interrupted["returncode"] if interrupted else None,
        status=("interrupted" if interrupted["returncode"] else "completed_before_timeout") if interrupted else "before_full_copy",
        disk=shutil.disk_usage(args.destination.parent)._asdict(),
    ))
    if interrupted is None or interrupted["returncode"]:
        resumed = download(args.drive_path, args.destination, args.receipts,
                           attempt="download-resumed", seconds=10800, resume=args.destination.exists())
        if resumed["returncode"]:
            raise RuntimeError("Download resume failed; retain the partial copy and receipts for inspection")
    else:
        resumed = None
    disk_downloaded = shutil.disk_usage(args.destination)._asdict()
    if dual_sha256(args.destination / "distribution.json") != args.manifest_sha256:
        raise ValueError("Downloaded distribution manifest differs from the pinned local manifest")
    partial_recovery = reconcile_partial(args.destination, partial, manifest)
    write(args.receipts / "partial-recovery.json", partial_recovery)
    verify_started = time.monotonic()
    verification = verify_distribution(args.destination, workers=args.hash_workers)
    verification_seconds = time.monotonic() - verify_started
    write(args.receipts / "file-verification.json", verification)
    sample_started = time.monotonic()
    windows = [tuple(value) for value in json.loads(args.windows.read_text())]
    expected = json.loads(args.reference.read_text())
    actual = sample_windows(BallFrameStore(args.destination), windows)
    write(args.receipts / "samples.json", actual)
    if actual != expected:
        raise ValueError("Colab and local sample teacher inputs/RGB/JPEG/native PTS differ")
    summary = dict(
        schema_version=1, status="verified", fresh_destination=not args.resume_existing,
        interrupted_download=interrupted, resumed_download=resumed,
        manifest_download=manifest_fetch, partial_recovery=partial_recovery,
        download_seconds=manifest_fetch["elapsed_seconds"] + (interrupted["elapsed_seconds"] if interrupted else 0) + (resumed["elapsed_seconds"] if resumed else 0),
        sha256_workers=args.hash_workers,
        sha256_verification_seconds=verification_seconds, sample_seconds=time.monotonic() - sample_started,
        total_seconds=time.monotonic() - started, samples_equal=True,
        file_verification=verification, disk_after_download=disk_downloaded,
        disk_after_verification=shutil.disk_usage(args.destination)._asdict(),
        reference_sha256=dual_sha256(args.reference), sampled_windows=len(windows),
    )
    write(args.receipts / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-path", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-size-bytes", type=int, required=True)
    parser.add_argument("--hash-workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interrupt-download", action="store_true")
    mode.add_argument("--resume-existing", action="store_true")
    run(parser.parse_args())
