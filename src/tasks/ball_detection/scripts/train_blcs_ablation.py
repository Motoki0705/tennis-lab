"""Run a fixed-update BLCS loss ablation through the shared training queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.blcs.ablation.cli_paths import experiment_paths
from src.tasks.blcs.ablation.progressive_losses import ARMS, run_experiment
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="blcs.ablation.train",
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
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-dir", help="Dataset directory relative to --data-root")
    parser.add_argument(
        "--output", required=True, help="Child of the checkout outputs root"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--gan", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_absolute():
        parser.error("--data-root must be an explicit absolute path")
    if args.steps <= 0 or (args.gan and not args.smoke):
        parser.error("Require positive updates; GAN is only supported for smoke runs")
    resolver = experiment_paths(
        data_root=args.data_root, output_root=PROJECT_ROOT / "outputs"
    )
    output = resolver.resolve(PathRole.OUTPUT, args.output)
    paths = PATH_BOUNDARY.validate(
        {"data_root": args.data_root, "output": output}, resolver=resolver
    )
    args.data_root = paths.declared("data_root").path
    run_experiment(args)


if __name__ == "__main__":
    main()
