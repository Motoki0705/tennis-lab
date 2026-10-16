"""Explicitly migrate a sparse v1 player frame store into a new v2 store."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.player_detection.data.migration import migrate_v1_store
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
    name="player_detection.migrate_store",
    fields=(
        BoundaryPathField(
            "source", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY,
            must_exist=True, allow_role_root=True,
        ),
        BoundaryPathField(
            "destination", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing v1 store (absolute path)")
    parser.add_argument("destination", type=Path, help="New v2 store (absolute path)")
    args = parser.parse_args()
    if not args.source.is_absolute() or not args.destination.is_absolute():
        raise PathContractError("Migration source and destination must be absolute paths")
    source, destination = args.source.resolve(), args.destination.resolve()
    # Each explicit store grants only its own read/write scope, even when the
    # stores live under unrelated directories. No common ancestor is granted.
    roots = RuntimePathRoots(
        project_root=source, data_root=source, external_asset_root=source,
        output_root=destination, artifact_root=destination,
        checkpoint_root=destination, cache_root=destination,
    )
    paths = PATH_BOUNDARY.validate(
        {"source": args.source, "destination": args.destination},
        resolver=PathResolver(roots),
    )
    print(migrate_v1_store(
        paths.declared("source").path, paths.declared("destination").path,
    ))


if __name__ == "__main__":
    main()
