"""Generate a CPU-only, immutable 3D-refiner synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import load_plan
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import generate_dataset
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
    name="ball_refiner.generate_synthetic_3d",
    fields=(
        BoundaryPathField("plan", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.DATA, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "project-root", "data-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.plan, args.output, args.project_root, args.data_root)):
        parser.error("All paths must be absolute")
    roots = RuntimePathRoots(
        project_root=args.project_root, data_root=args.data_root,
        artifact_root=args.output.parent, output_root=args.output.parent,
        checkpoint_root=args.output.parent, cache_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    resolver = PathResolver(roots)
    paths = PATH_BOUNDARY.validate({"plan": args.plan, "output": args.output}, resolver=resolver)
    plan = load_plan(paths.declared("plan").path, resolver)
    generate_dataset(plan, paths.declared("output").path, mode=args.mode)


if __name__ == "__main__":
    main()
