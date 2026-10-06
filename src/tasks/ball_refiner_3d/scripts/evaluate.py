"""Evaluate an existing checkpoint with its recorded test corruption and seed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from src.tasks.ball_refiner_3d.evaluation.offline import evaluate_run
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner_3d.evaluate",
    fields=(
        BoundaryPathField(
            "checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "run_config",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "dataset",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "run-config", "dataset", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    values = {
        key: getattr(args, key)
        for key in ("checkpoint", "run_config", "dataset", "output")
    }
    if not all(path.is_absolute() for path in values.values()):
        parser.error("All paths must be explicit absolute paths")
    if args.device == "cuda" and not os.environ.get("TENNIS_RUN_ID"):
        parser.error(
            "CUDA evaluation must be launched through the shared training queue"
        )
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        checkpoint_root=args.checkpoint.parent,
        artifact_root=args.run_config.parent,
        data_root=args.dataset,
        output_root=args.output.parent,
        cache_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(values, resolver=PathResolver(roots))
    torch.set_num_threads(2)
    evaluate_run(
        **{key: paths.declared(key).path for key in values},
        device=args.device,
        batch_size=args.batch_size,
    )
    print(args.output)


if __name__ == "__main__":
    main()
