"""Receipt-backed file-level upload/resume using only the tennis-drive CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.distribution import load_distribution
from src.tasks.ball_detection.data.distribution.manifest import distribution_files
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version

DRIVE_CLI = Path(".agents/skills/tennis-drive/scripts/drive.py")


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def call(receipts: Path, label: str, arguments: list[str], *, timeout: int = 1800) -> dict[str, Any]:
    stem = receipts / (label + "-" + uuid.uuid4().hex[:10])
    argv = [sys.executable, str(DRIVE_CLI), "--timeout-seconds", str(timeout), *arguments]
    started_at, started = datetime.now(UTC).isoformat(), time.monotonic()
    with stem.with_suffix(".stdout.json").open("wb") as stdout, stem.with_suffix(".stderr.log").open("wb") as stderr:
        process = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
    receipt = dict(schema_version=1, argv=argv, started_at=started_at,
                   elapsed_seconds=time.monotonic() - started, returncode=process.returncode,
                   stdout=stem.with_suffix(".stdout.json").name, stderr=stem.with_suffix(".stderr.log").name)
    save(stem.with_suffix(".timing.json"), receipt)
    return receipt


def checked(receipts: Path, label: str, arguments: list[str]) -> dict[str, Any]:
    receipt = call(receipts, label, arguments)
    if receipt["returncode"]:
        raise RuntimeError(f"Drive command failed; inspect {receipt['stdout']} and {receipt['stderr']}")
    return receipt


def inventory(receipts: Path, remote: str) -> dict[str, Any]:
    output = receipts / ("inventory-" + uuid.uuid4().hex[:10] + ".json")
    checked(receipts, "inventory", ["inventory", "--path", remote, "--output", str(output)])
    result: dict[str, Any] = json.loads(output.read_text())
    if result["ambiguous_paths"]:
        raise ValueError("Duplicate Drive paths; never deduplicate automatically")
    return result


def fingerprints(files: dict[str, Path], expected: dict[str, dict[str, Any]], receipts: Path) -> dict[str, Any]:
    # Rehash on every resume. Filesystem timestamp granularity can conceal a
    # rapid same-size rewrite, so an old stat tuple is insufficient evidence.
    result = {}
    progress_at = time.monotonic()
    for name, path in files.items():
        before = file_version(path)
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha.update(data)
                md5.update(data)
        if (file_version(path) != before or before[2] != expected[name]["size_bytes"]
                or sha.hexdigest() != expected[name]["sha256"]):
            raise ValueError(f"Upload source differs from its immutable distribution: {name}")
        result[name] = dict(size_bytes=before[2], sha256=sha.hexdigest(), md5=md5.hexdigest(), stat=list(before))
        if time.monotonic() - progress_at >= 30:
            print(json.dumps(dict(status="hashing_sources", files=len(result), total_files=len(files), path=name)), flush=True)
            progress_at = time.monotonic()
    save(receipts / "file-fingerprints.json", result)
    return result


def matching(document: dict[str, Any], expected: dict[str, Any]) -> set[str]:
    entries = {e["path"]: e for e in document["entries"] if e["type"] == "file"}
    if set(entries) - set(expected):
        raise ValueError("Drive contains files outside this distribution")
    result = set()
    for name, entry in entries.items():
        if (entry["size_bytes"] != expected[name]["size_bytes"]
                or entry["hashes"].get("md5") != expected[name]["md5"]):
            raise ValueError(f"Existing Drive content differs: {name}; inspect partial-copy receipts before repair")
        result.add(name)
    return result


def transfer(args: argparse.Namespace) -> None:
    if not os.environ.get("RCLONE_CONFIG"):
        raise ValueError("Select the authenticated Drive configuration with RCLONE_CONFIG")
    if not args.drive_path.startswith("data/ball_detection/") or args.workers not in range(1, 5):
        raise ValueError("Use a versioned ball input destination and one to four upload workers")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("A controlled partial copy must select at least one file")
    args.receipts.mkdir(parents=True, exist_ok=True)
    started_at, started = datetime.now(UTC).isoformat(), time.monotonic()
    manifest = load_distribution(args.distribution)
    metadata = json.loads((args.distribution / "metadata.json").read_text())
    if args.drive_path.rsplit("/", 1)[-1] != metadata["version"]:
        raise ValueError("Drive input directory must use this distribution's explicit version")
    identity = dual_sha256(args.distribution / "distribution.json")
    plan = dict(schema_version=1, drive_path=args.drive_path, distribution_sha256=identity)
    state = args.receipts / "transfer-plan.json"
    owned = state.exists()
    if owned and json.loads(state.read_text()) != plan:
        raise ValueError("Resume requires the exact original transfer plan")
    files = distribution_files(args.distribution, source_directory=args.source_store)
    files["distribution.json"] = args.distribution / "distribution.json"
    expected = {e["path"]: e for e in manifest["files"]}
    expected["distribution.json"] = dict(size_bytes=files["distribution.json"].stat().st_size, sha256=identity)
    hash_started = time.monotonic()
    hashes = fingerprints(files, expected, args.receipts)
    hash_seconds = time.monotonic() - hash_started
    checked(args.receipts, "mkdir", ["mkdir", args.drive_path])
    before = inventory(args.receipts, args.drive_path)
    if not owned and before["entries"]:
        raise ValueError("A new transfer cannot adopt an existing Drive directory")
    complete = matching(before, hashes)
    if not owned:
        save(state, plan)
    payloads = [c["payload"] for c in manifest["clips"]]
    pending = [name for name in payloads if name not in complete]
    if args.interrupt_largest:
        if not pending:
            raise ValueError("No missing payload is available for the planned interruption")
        name = max(pending, key=lambda name: expected[name]["size_bytes"])
        checked(args.receipts, "mkdir-shards", ["mkdir", args.drive_path + "/shards"])
        receipt = call(args.receipts, "interrupted-upload", ["upload", str(files[name]), args.drive_path + "/" + name, "--verify"], timeout=20)
        after = inventory(args.receipts, args.drive_path)
        matched = matching(after, hashes)
        save(args.receipts / "interruption.json", dict(
            schema_version=1, file=name, size_bytes=expected[name]["size_bytes"],
            source_hash_seconds=hash_seconds,
            receipt=receipt, completed_files=sorted(matched), status="interrupted" if receipt["returncode"] else "completed_before_timeout",
            resume_unit="whole uncompleted file; previously verified files are retained",
        ))
        print(json.dumps(dict(status="interruption_recorded", file=name, returncode=receipt["returncode"])), flush=True)
        return
    if args.max_files is not None:
        pending = pending[:args.max_files]
    lock, stop = threading.Lock(), threading.Event()
    remaining = iter(pending)
    completed, failures = [], []

    def worker() -> None:
        while not stop.is_set():
            with lock:
                name = next(remaining, None)
            if name is None:
                return
            receipt = call(args.receipts, "payload-upload", ["upload", str(files[name]), args.drive_path + "/" + name, "--verify"])
            if receipt["returncode"] or file_version(files[name]) != tuple(hashes[name]["stat"]):
                with lock:
                    failures.append(name)
                stop.set()
                return
            with lock:
                completed.append(name)
                print(json.dumps(dict(status="file_verified", completed_this_attempt=len(completed), total_this_attempt=len(pending), path=name)), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.workers)]
        for future in futures:
            future.result()
    current = inventory(args.receipts, args.drive_path)
    complete = matching(current, hashes)
    summary = dict(schema_version=1, started_at=started_at, elapsed_seconds=time.monotonic() - started,
                   source_hash_seconds=hash_seconds,
                   distribution_sha256=identity, completed_before=len(matching(before, hashes)),
                   uploaded_payloads=len(completed), failures=failures,
                   completed_files=len(complete), expected_files=len(files),
                   completed_bytes=sum(hashes[name]["size_bytes"] for name in complete),
                   expected_bytes=sum(e["size_bytes"] for e in hashes.values()))
    if failures:
        summary["status"] = "failed"
    elif args.max_files is not None:
        summary["status"] = "paused_between_files"
    else:
        if not set(payloads) <= complete:
            raise ValueError("Payloads are not complete; refuse to publish metadata")
        small = set(files) - set(payloads) - {"distribution.json"}
        if not small <= complete:
            bundle = args.receipts / "metadata-source"
            if bundle.exists():
                shutil.rmtree(bundle)
            bundle.mkdir()
            for name in sorted(small):
                target = bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(files[name], target)
            checked(args.receipts, "metadata-upload", ["upload", str(bundle), args.drive_path, "--overwrite", "--verify"])
        check = matching(inventory(args.receipts, args.drive_path), hashes)
        if not set(files) - {"distribution.json"} <= check:
            raise ValueError("Refuse to publish a manifest before every input is verified")
        if "distribution.json" not in check:
            checked(args.receipts, "manifest-upload", ["upload", str(files["distribution.json"]), args.drive_path + "/distribution.json", "--verify"])
        final = inventory(args.receipts, args.drive_path)
        if matching(final, hashes) != set(files):
            raise ValueError("Final Drive inventory differs from the distribution")
        summary.update(status="verified", completed_files=len(files), completed_bytes=summary["expected_bytes"],
                       final_inventory=final, elapsed_seconds=time.monotonic() - started)
    save(args.receipts / ("attempt-" + uuid.uuid4().hex[:10] + ".json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "final_inventory"}), flush=True)
    if failures:
        raise RuntimeError("Upload failed; inspect receipts and inventory before resuming")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--drive-path", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--max-files", type=int)
    modes.add_argument("--interrupt-largest", action="store_true")
    transfer(parser.parse_args())
