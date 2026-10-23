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


def run(args: argparse.Namespace) -> None:
    if not os.environ.get("RCLONE_CONFIG"):
        raise ValueError("The Colab worker must supply its managed Drive configuration")
    if args.destination.exists() or args.receipts.exists():
        raise FileExistsError("Fresh-VM verification requires new destination and receipt directories")
    args.receipts.mkdir(parents=True)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write(args.receipts / "environment.json", dict(
        schema_version=1, observed_at=datetime.now(UTC).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=sys.version, disk_before_download=shutil.disk_usage(args.destination.parent)._asdict(),
        uv_lock_sha256=dual_sha256(Path("uv.lock")),
    ))
    if shutil.disk_usage(args.destination.parent).free < args.expected_size_bytes:
        raise ValueError("Fresh VM has insufficient free space for the fixed distribution")
    # The Drive command owns rclone's timeout. A partial local directory is
    # inventoried and checked before --overwrite is used to resume that copy.
    interrupted = download(args.drive_path, args.destination, args.receipts,
                           attempt="download-interrupted", seconds=45, resume=False)
    partial = [dict(path=p.relative_to(args.destination).as_posix(), size_bytes=p.stat().st_size,
                    sha256=dual_sha256(p)) for p in args.destination.rglob("*") if p.is_file()]
    write(args.receipts / "download-partial.json", dict(
        schema_version=1, files=partial, returncode=interrupted["returncode"],
        status="interrupted" if interrupted["returncode"] else "completed_before_timeout",
        disk=shutil.disk_usage(args.destination.parent)._asdict(),
    ))
    if interrupted["returncode"]:
        resumed = download(args.drive_path, args.destination, args.receipts,
                           attempt="download-resumed", seconds=10800, resume=args.destination.exists())
        if resumed["returncode"]:
            raise RuntimeError("Download resume failed; retain the partial copy and receipts for inspection")
    else:
        resumed = None
    disk_downloaded = shutil.disk_usage(args.destination)._asdict()
    if dual_sha256(args.destination / "distribution.json") != args.manifest_sha256:
        raise ValueError("Downloaded distribution manifest differs from the pinned local manifest")
    verify_started = time.monotonic()
    verification = verify_distribution(args.destination)
    verification_seconds = time.monotonic() - verify_started
    write(args.receipts / "file-verification.json", verification)
    sample_started = time.monotonic()
    windows = [tuple(value) for value in json.loads(args.windows.read_text())]
    expected = json.loads(args.reference.read_text())
    actual = sample_windows(BallFrameStore(args.destination), windows)
    write(args.receipts / "samples.json", actual)
    if actual != expected:
        raise ValueError("Colab and local sample teacher inputs/RGB/JPEG/native PTS differ")
    for entry in partial:
        path = args.destination / entry["path"]
        # Rclone temporary files need not survive; any completed file preserved
        # by the resume must match its original content, not just its size.
        if path.is_file() and not path.name.endswith(".partial") and dual_sha256(path) != entry["sha256"]:
            raise ValueError("A completed partial-copy file changed across resume")
    summary = dict(
        schema_version=1, status="verified", fresh_destination=True,
        interrupted_download=interrupted, resumed_download=resumed,
        download_seconds=interrupted["elapsed_seconds"] + (resumed["elapsed_seconds"] if resumed else 0),
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
    run(parser.parse_args())
