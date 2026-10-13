"""Evaluate the ten completed BLCS runs and fixed references."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.blcs.experiments.cli_paths import experiment_paths
from src.tasks.blcs.experiments.final_evaluation import run_final_evaluation
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="blcs.issue1044_final_evaluation",
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
            "historical_checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "real_store",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "run_root",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--real-store", type=Path, required=True)
    args = parser.parse_args()
    args.run_root = PROJECT_ROOT / "outputs/blcs/i1044"
    values = {
        key: getattr(args, key)
        for key in ("data_root", "historical_checkpoint", "real_store", "run_root")
    }
    if not all(path.is_absolute() for path in values.values()):
        parser.error("All paths must be explicit absolute paths")
    paths = PATH_BOUNDARY.validate(
        values,
        resolver=experiment_paths(
            data_root=args.data_root,
            output_root=PROJECT_ROOT / "outputs",
            checkpoint=args.historical_checkpoint,
            artifact_root=args.real_store.parent,
        ),
    )
    for key in values:
        setattr(args, key, paths.declared(key).path)
    run_final_evaluation(args)


if __name__ == "__main__":
    main()
