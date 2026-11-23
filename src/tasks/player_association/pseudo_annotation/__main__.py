"""Extract, prepare, annotate or publish partial player association labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.tasks.player_association.pseudo_annotation.calibration import (
    prepare_calibration,
)
from src.tasks.player_association.pseudo_annotation.extract import extract, load_plan
from src.tasks.player_association.pseudo_annotation.frozen_run import FrozenRun
from src.tasks.player_association.pseudo_annotation.prepare import prepare_baseline
from src.tasks.player_association.pseudo_annotation.runner import (
    publish,
    run_annotation,
)
from src.tasks.player_association.pseudo_annotation.sampling import STRATA
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.io import write_json_atomic
from src.utils.paths import PROJECT_ROOT

_CONFIG = (
    PROJECT_ROOT / "src/tasks/player_association/configs/pseudo_annotation_meiji.yaml"
)
PATH_BOUNDARY = NonHydraPathBoundary(
    name="player_association.pseudo_annotation",
    fields=(
        BoundaryPathField(
            "snapshot",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "config",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "plan",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "packet",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "run_dir",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "dataset",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "destination",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.FILE,
            allow_role_root=True,
            required=False,
        ),
        BoundaryPathField(
            "output_dir",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
            required=False,
        ),
    ),
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="action", required=True)
    plan = command.add_parser("extract")
    plan.add_argument("snapshot", type=Path)
    plan.add_argument("destination", type=Path)
    plan.add_argument("--config", type=Path, default=_CONFIG)
    prepare = command.add_parser("prepare")
    prepare.add_argument("snapshot", type=Path)
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("clip_id")
    prepare.add_argument("output_dir", metavar="packet", type=Path)
    subset = prepare.add_mutually_exclusive_group()
    subset.add_argument("--frames", type=int, nargs="+")
    subset.add_argument("--stratum", choices=STRATA)
    prepare.add_argument("--show-prior-association", action="store_true")
    calibration = command.add_parser("prepare-calibration")
    calibration.add_argument("dataset", type=Path)
    calibration.add_argument("clip_id")
    calibration.add_argument("output_dir", metavar="packet", type=Path)
    calibration.add_argument("--frames", type=int, nargs="+", required=True)
    run = command.add_parser("run")
    run.add_argument("packet", type=Path)
    run.add_argument("output_dir", metavar="run_dir", type=Path)
    store = command.add_parser("publish")
    store.add_argument("packet", type=Path)
    store.add_argument("run_dir", type=Path)
    store.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    arguments = {
        name: value.expanduser().absolute()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    output = next(
        (
            arguments[field.name]
            for field in PATH_BOUNDARY.fields
            if field.direction is PathDirection.OUTPUT and field.name in arguments
        ),
        None,
    )
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=arguments["dataset"].resolve()
        if "dataset" in arguments
        else PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=PROJECT_ROOT
        if output is None
        else output.parent.resolve() / output.name,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        arguments,
        resolver=PathResolver(roots),
        independent_artifact_inputs=True,
    )
    if args.action == "extract":
        result = extract(
            FrozenRun(paths.declared("snapshot").path), paths.declared("config").path
        )
        write_json_atomic(paths.declared("destination").path, result)
        print(json.dumps(result["summary"], indent=2))
    elif args.action == "prepare":
        frozen = FrozenRun(paths.declared("snapshot").path)
        plan_data = load_plan(paths.declared("plan").path, frozen)
        scope = plan_data["clips"][args.clip_id]
        frames = (
            args.frames
            if args.frames is not None
            else (
                scope["sampled"][args.stratum]
                if args.stratum
                else scope["sampled_frames"]
            )
        )
        prepare_baseline(
            frozen,
            plan_data,
            args.clip_id,
            frames,
            paths.declared("output_dir").path,
            show_prior_association=args.show_prior_association,
        )
    elif args.action == "prepare-calibration":
        prepare_calibration(
            paths.declared("dataset").path,
            args.clip_id,
            args.frames,
            paths.declared("output_dir").path,
        )
    elif args.action == "run":
        receipt = run_annotation(
            paths.declared("packet").path, paths.declared("output_dir").path
        )
        print(
            json.dumps(
                {key: value for key, value in receipt.items() if key != "command"},
                indent=2,
            )
        )
    else:
        print(
            publish(
                paths.declared("packet").path,
                paths.declared("run_dir").path,
                paths.declared("dataset").path,
            )
        )


if __name__ == "__main__":
    main()
