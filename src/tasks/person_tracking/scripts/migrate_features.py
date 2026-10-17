"""Explicitly migrate a v2 person feature archive into a new v3 archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.person_tracking.migration import migrate_v2_features
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathContractError,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="person_tracking.migrate_features",
    fields=(
        BoundaryPathField(
            "source", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
            must_exist=True, allow_role_root=True,
        ),
        BoundaryPathField(
            "destination", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing v2 archive (absolute path)")
    parser.add_argument("destination", type=Path, help="New v3 archive (absolute path)")
    args = parser.parse_args()
    if not args.source.is_absolute() or not args.destination.is_absolute():
        raise PathContractError("Migration source and destination must be absolute paths")
    source, destination = args.source.resolve(), args.destination.resolve()
    # Grant only the explicit input file and output file, not a common ancestor.
    roots = RuntimePathRoots(
        project_root=source, data_root=source, external_asset_root=source,
        artifact_root=source, output_root=destination,
        checkpoint_root=destination, cache_root=destination,
    )
    paths = PATH_BOUNDARY.validate(
        {"source": args.source, "destination": args.destination},
        resolver=PathResolver(roots),
    )
    # Check the caller's path too: resolution would hide a dangling symlink.
    if args.destination.exists() or args.destination.is_symlink():
        raise FileExistsError(f"Migration destination already exists: {args.destination}")
    print(migrate_v2_features(
        paths.declared("source").path, paths.declared("destination").path,
    ))


if __name__ == "__main__":
    main()
