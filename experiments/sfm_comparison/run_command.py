"""Run one bounded queue job and retain its actual outputs for comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def numeric_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Export the flat numeric mapping expected by knowledge-control."""
    values = {"elapsed_seconds": float(record["elapsed_seconds"])}
    peak = record.get("device_peak_memory_mib")
    if peak is not None:
        values["device_peak_memory_mib"] = float(peak)

    def collect(prefix: str, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if key in {"gates", "minimum_supported_points_per_image"}:
                continue  # Thresholds and gate policy are not measured scores.
            name = f"{prefix}/{key}"
            if isinstance(value, dict):
                collect(name, value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values[name] = float(value)

    collect("sfm", record["metrics"])
    return values


def sample_memory(stop: threading.Event, samples: list[float]) -> None:
    """Measure whole-device memory, explicitly not process-only allocations."""
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            values = result.stdout.strip().splitlines()
            if len(values) == 1:
                samples.append(float(values[0]))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # A missing measurement remains null, never an invented zero.
        stop.wait(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("sfm", "frontend", "sanity"), required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--metrics-source", type=Path)
    parser.add_argument("--required-path", type=Path, required=True)
    parser.add_argument("--no-memory-sampling", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        parser.error("Experiment execution must be launched by training-queue")
    if args.seconds <= 0 or args.seconds > 21600:
        parser.error("Run timeout must be in 1..21600 seconds")
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a single directory name")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Missing experiment command")
    output = args.campaign.resolve() / "runs" / args.run_id
    if output.exists():
        parser.error("Run output already exists; use a distinct attempt ID")
    output.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    record: dict[str, Any] = {
        "run_id": args.run_id,
        "queue_job_id": os.environ["TENNIS_RUN_ID"],
        "stage": args.stage,
        "cwd": str(Path.cwd()),
        "command": command,
        "timeout_seconds": args.seconds,
        "started_at": started,
        "status": "running",
        "required_path": str(args.required_path.resolve()),
    }
    write_json(output / "execution.json", record)
    stop = threading.Event()
    samples: list[float] = []
    monitor = threading.Thread(target=sample_memory, args=(stop, samples), daemon=True)
    if not args.no_memory_sampling:
        monitor.start()
    start = time.monotonic()
    try:
        with (output / "command.log").open("w") as stream:
            # --foreground keeps all work inside the queue-created process group.
            completed = subprocess.run(
                [
                    "timeout",
                    "--foreground",
                    "--signal=TERM",
                    "--kill-after=30s",
                    str(args.seconds),
                    *command,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = completed.returncode
    except OSError as error:
        record["launch_error"] = str(error)
        returncode = 127
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join(timeout=6)
    elapsed = time.monotonic() - start
    record.update(
        {
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "required_path_exists": args.required_path.exists(),
            "device_memory_sampling": "whole-device MiB at approximately 2-second intervals",
            "device_memory_sample_count": len(samples),
            "device_peak_memory_mib": max(samples) if samples else None,
            "metrics": {},
        }
    )
    if args.metrics_source is not None and args.metrics_source.is_file():
        try:
            metrics = json.loads(args.metrics_source.read_text())
            if not isinstance(metrics, dict):
                raise ValueError("Expected a metrics mapping")
            json.dumps(metrics, allow_nan=False)
            record["metrics"] = metrics
            record["metrics_source"] = str(args.metrics_source.resolve())
        except (ValueError, OSError) as error:
            record["metrics_error"] = str(error)
    missing_metrics = args.metrics_source is not None and "metrics_source" not in record
    record["status"] = (
        "done"
        if returncode == 0 and args.required_path.exists() and not missing_metrics
        else "failed"
    )
    write_json(output / "execution.json", record)
    repro = Path(os.environ["TENNIS_REPRO_DIR"])
    write_json(repro / "execution.json", record)
    write_json(repro / "predictions/metrics.json", numeric_metrics(record))
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": record["status"],
                "elapsed_seconds": elapsed,
                "output": str(output),
            }
        )
    )
    return 0 if record["status"] == "done" else (returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
