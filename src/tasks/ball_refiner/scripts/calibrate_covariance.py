"""CPU-only clip cross-validation of a saved refiner covariance scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.tasks.ball_refiner.evaluation.covariance_calibration import (
    run_covariance_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--bounds", type=float, nargs=2, required=True)
    parser.add_argument("--grid-points", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 4:
        parser.error("Require 1–4 CPU threads")
    if not all(p.is_absolute() for p in (args.predictions, args.output, args.calibration_artifact)):
        parser.error("All paths must be absolute")
    torch.set_num_threads(args.cpu_threads)
    run_covariance_calibration(args.predictions, args.output, args.calibration_artifact,
                              bounds=tuple(args.bounds), grid_points=args.grid_points)


if __name__ == "__main__":
    main()
