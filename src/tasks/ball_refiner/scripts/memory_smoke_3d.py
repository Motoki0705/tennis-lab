"""Explicit analytic-input memory smoke for the 3D diffusion scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.diffusion.memory_smoke import run_memory_smoke
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
    name="ball_refiner.memory_smoke_3d",
    fields=(
        BoundaryPathField("config", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("fixture", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "fixture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.config, args.fixture, args.output)):
        parser.error("All paths must be absolute")
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.output.parent,
        artifact_root=args.config.parent, output_root=args.output.parent,
        checkpoint_root=args.output.parent, cache_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate({"config": args.config, "fixture": args.fixture, "output": args.output}, resolver=PathResolver(roots))
    run_memory_smoke(paths.declared("config").path, paths.declared("fixture").path, paths.declared("output").path, device=args.device)


if __name__ == "__main__":
    main()
