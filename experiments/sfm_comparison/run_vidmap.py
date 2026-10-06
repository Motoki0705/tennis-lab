"""Run the official VidMap CLI, then validate and measure its saved reconstruction."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.sfm_comparison.evaluate_model import evaluate
from experiments.sfm_comparison.run_command import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        parser.error("VidMap must run inside a training-queue job")
    if args.output.exists():
        parser.error("Output exists; use a distinct attempt directory")
    start = time.monotonic()
    # Keep upstream's native-before-Torch import order in a fresh process.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vidmap.run",
            "--input_data",
            str(args.images),
            "--output",
            str(args.output),
            "--device",
            "cuda",
        ],
        check=True,
    )
    mapped = time.monotonic()
    evaluate(
        args.output / "rec",
        args.manifest,
        args.images,
        args.evaluator_root,
        args.output / "evaluation",
    )
    write_json(
        args.output / "phase-times.json",
        {
            "vidmap_seconds": mapped - start,
            "cpu_evaluation_seconds": time.monotonic() - mapped,
            "note": "Cold/warm compilation and model caches are not controlled speed benchmarks.",
        },
    )


if __name__ == "__main__":
    main()
