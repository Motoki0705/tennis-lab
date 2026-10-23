"""Time an unmodified CLI and sample its device-wide GPU usage."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv or args.receipt.exists():
        raise ValueError("Supply a command and a new receipt path")
    done = threading.Event()
    samples: list[tuple[float, int, int]] = []
    errors: list[str] = []

    def monitor() -> None:
        while not done.is_set():
            try:
                value = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
                memory, utilization = [int(item.strip()) for item in value.split(",")]
                samples.append((time.perf_counter(), memory, utilization))
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                return
            done.wait(0.5)

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    if args.sample_gpu:
        thread.start()
    result = None
    try:
        result = subprocess.run(argv, check=False)
    finally:
        finished = time.perf_counter()
        done.set()
        if args.sample_gpu:
            thread.join(timeout=12)
        receipt = {
            "schema": "court_pilot_command_measurement_v1",
            "argv": argv,
            "started_at": started_at,
            "wall_seconds": finished - started,
            "returncode": None if result is None else result.returncode,
            "sampled_gpu": args.sample_gpu,
            "gpu_samples": len(samples),
            "gpu_peak_used_mib": max((item[1] for item in samples), default=None),
            "gpu_utilization_mean_percent": sum(item[2] for item in samples)
            / len(samples)
            if samples
            else None,
            "gpu_sampling_errors": errors,
            "gpu_memory_method": "device-wide nvidia-smi every 0.5 seconds; sampled lower bound",
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_suffix(".partial")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.receipt)
    if errors:
        raise RuntimeError(f"GPU sampling failed: {errors}")
    if result is None or result.returncode:
        raise SystemExit(1 if result is None else result.returncode)


if __name__ == "__main__":
    main()
