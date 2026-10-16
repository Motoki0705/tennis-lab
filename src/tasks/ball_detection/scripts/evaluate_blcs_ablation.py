"""Evaluate a frozen BLCS checkpoint or a fixed diagnostic control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.tasks.blcs.ablation.cli_paths import experiment_paths
from src.tasks.blcs.ablation.evaluation import evaluate
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="blcs.ablation.evaluation",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
        BoundaryPathField(
            "checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            required=False,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path)
    choice.add_argument(
        "--baseline",
        choices=("gt", "geometry", "jitter", "time_shift", "collision_crossing"),
    )
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fitting-clips", type=int, default=12)
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or args.fitting_clips < 0:
        parser.error("--limit must be positive and --fitting-clips nonnegative")
    if args.historical and args.checkpoint is None:
        parser.error("--historical requires --checkpoint")
    values = {"data_root": args.data_root, "output": args.output}
    if args.checkpoint is not None:
        values["checkpoint"] = args.checkpoint
    if not all(path.is_absolute() for path in values.values()):
        parser.error("All paths must be explicit absolute paths")
    paths = PATH_BOUNDARY.validate(
        values,
        resolver=experiment_paths(
            data_root=args.data_root,
            output_root=args.output.parent,
            checkpoint=args.checkpoint,
        ),
    )
    if not os.environ.get("TENNIS_RUN_ID"):
        parser.error("Evaluation must run through the shared training queue")
    torch.set_num_threads(2)
    result = evaluate(
        data_root=paths.declared("data_root").path,
        output=paths.declared("output").path,
        split=args.split,
        checkpoint=paths.declared("checkpoint").path if args.checkpoint else None,
        baseline=args.baseline,
        historical=args.historical,
        limit=args.limit,
        fitting_clips=args.fitting_clips,
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "trajectory_rmse_m": result["metrics"]["trajectory_rmse_m"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
