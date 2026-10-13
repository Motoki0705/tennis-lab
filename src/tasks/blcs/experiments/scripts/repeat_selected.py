"""Freeze validation-only selection and run its additional seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.tasks.blcs.experiments.cli_paths import experiment_paths
from src.tasks.blcs.experiments.issue1044 import run_experiment
from src.tasks.blcs.experiments.repeat_selected import select_additional_arm
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="blcs.issue1044_repeat",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
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
    parser.add_argument("--seed", type=int, choices=(43, 44))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if args.select_only:
        if args.seed is not None or args.data_root is not None:
            parser.error("--select-only cannot be combined with training arguments")
    elif args.seed is None or args.data_root is None:
        parser.error("Training requires --seed and --data-root")
    root = PROJECT_ROOT / "outputs/blcs/i1044"
    values = {"run_root": root}
    if args.data_root is not None:
        values["data_root"] = args.data_root
        if not args.data_root.is_absolute():
            parser.error("--data-root must be an explicit absolute path")
    paths = PATH_BOUNDARY.validate(
        values,
        resolver=experiment_paths(
            data_root=args.data_root if args.data_root else PROJECT_ROOT / "data",
            output_root=PROJECT_ROOT / "outputs",
        ),
    )
    if not os.environ.get("TENNIS_RUN_ID"):
        parser.error("Additional training must run through the shared queue")
    selection = select_additional_arm(paths.declared("run_root").path)
    print(json.dumps(selection), flush=True)
    if args.select_only:
        return
    arm = selection["selected_arm"]
    run_experiment(
        argparse.Namespace(
            arm=arm,
            seed=args.seed,
            steps=4000,
            data_root=paths.declared("data_root").path,
            output=f"blcs/i1044/{arm}-s{args.seed}-u4000",
            smoke=False,
            chunked=False,
            gan=False,
        )
    )


if __name__ == "__main__":
    main()
