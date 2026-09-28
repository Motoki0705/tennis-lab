"""Export a completed detector-only pilot to a portable inference bundle (CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_refiner.deployment import export_pilot_bundle
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
    name="ball_refiner.export_pilot",
    fields=(
        BoundaryPathField("training_run", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.CHECKPOINT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.training_run, args.output)):
        parser.error("All paths must be absolute")
    roots = RuntimePathRoots(
        project_root=args.output.parent, data_root=args.output.parent, checkpoint_root=args.output.parent,
        cache_root=args.output.parent, artifact_root=args.training_run.parent,
        output_root=args.output.parent, external_asset_root=args.output.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"training_run": args.training_run, "output": args.output}, resolver=PathResolver(roots),
    )
    bundle = export_pilot_bundle(paths.declared("training_run").path, paths.declared("output").path)
    print(f"Published {bundle.directory}; manifest SHA-256 {bundle.manifest_sha256}")


if __name__ == "__main__":
    main()
