"""CPU-only clip cross-validation of a saved refiner covariance scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.tasks.ball_refiner.evaluation.covariance_calibration import (
    run_covariance_calibration,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.calibrate_covariance",
    fields=(
        BoundaryPathField("predictions", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
        BoundaryPathField("calibration_artifact", PathRole.CHECKPOINT, PathDirection.OUTPUT, PathKind.FILE),
    ),
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
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.output.parent,
        checkpoint_root=args.calibration_artifact.parent, cache_root=args.output.parent,
        artifact_root=args.predictions.parent, output_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"predictions": args.predictions, "output": args.output, "calibration_artifact": args.calibration_artifact},
        resolver=PathResolver(roots),
    )
    torch.set_num_threads(args.cpu_threads)
    run_covariance_calibration(paths.declared("predictions").path, paths.declared("output").path,
                              paths.declared("calibration_artifact").path,
                              bounds=tuple(args.bounds), grid_points=args.grid_points)


if __name__ == "__main__":
    main()
