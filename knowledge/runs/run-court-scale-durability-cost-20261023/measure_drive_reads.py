"""Time one existing cache receipt/archive through the unchanged Drive adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from src.synthetic_data_generation.dataset.court.scale.durable_storage import (
        BundleStore,
        SkillDriveFiles,
    )

    previous = root / "knowledge/runs/run-court-scale-colab-resume-l4-20261023"
    known = json.loads((previous / "interrupted-attempt.json").read_text())
    key = known["caches"][0]["cache_key"]
    drive_path = json.loads((previous / "drive-receipts/request.json").read_text())[
        "archive_drive"
    ]["path"].split("/objects/")[0]
    output = args.output.resolve()
    if not output.is_relative_to(root / "tmp"):
        raise ValueError("Keep this probe's scratch files inside this checkout's tmp/.")
    output.mkdir(parents=True, exist_ok=False)
    real = shutil.which("rclone")
    if real is None:
        raise ValueError("rclone is required.")
    wrapper = output / "rclone-trace.sh"
    tracer = Path(__file__).resolve().with_name("trace_rclone.py")
    wrapper.write_text(
        "#!/bin/sh\nexec "
        + shlex.join([sys.executable, str(tracer)])
        + ' "$@"\n'
    )
    wrapper.chmod(0o700)
    trace = output / "rclone-events.jsonl"
    os.environ.update(
        RCLONE_BIN=str(wrapper),
        COURT_COST_REAL_RCLONE=real,
        COURT_COST_TRACE=str(trace),
    )
    started_at = datetime.now(UTC).isoformat()
    store = BundleStore(output / "recovered", SkillDriveFiles(drive_path))
    started = time.perf_counter()
    receipt = store.receipt(f"receipts/caches/{key}.json")
    receipt_seconds = time.perf_counter() - started
    if receipt is None:
        raise ValueError("The existing smoke cache was not acknowledged.")
    expected = json.loads((previous / f"drive-receipts/caches/{key}.json").read_text())
    if receipt != expected:
        raise ValueError("Live receipt differs from the recorded smoke receipt.")
    started = time.perf_counter()
    store.restore(receipt)
    archive_seconds = time.perf_counter() - started
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    if any(event["returncode"] != 0 for event in events):
        raise ValueError("A traced rclone command failed.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[event["category"]].append(event)
    summary = {
        "schema": "court_scale_drive_read_cost_v1",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "probe_host": "local CPU; not Colab",
        "drive_path": drive_path,
        "cache_key": key,
        "archive_bytes": receipt["archive_bytes"],
        "receipt_seconds": receipt_seconds,
        "archive_restore_seconds": archive_seconds,
        "receipt_equals_recorded": True,
        "restored_file_count": len(receipt["files"]),
        "rclone_processes": len(events),
        "categories": {
            name: {
                "count": len(items),
                "seconds": sum(event["elapsed_seconds"] for event in items),
                "median_seconds": statistics.median(
                    event["elapsed_seconds"] for event in items
                ),
            }
            for name, items in sorted(groups.items())
        },
        "listed_parent_counts": dict(
            Counter(
                event["remote_target"]
                for event in events
                if event["category"] == "resolve_listing"
            )
        ),
        "tracer_sha256": hashlib.sha256(tracer.read_bytes()).hexdigest(),
        "http_request_count": None,
        "http_request_count_reason": "No credential/header logging; rclone can perform multiple HTTP calls per process.",
    }
    (output / "measurement.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
