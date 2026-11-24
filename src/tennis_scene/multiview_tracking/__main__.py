"""Run temporal 3D identity tracking on a saved frame pose candidate clip."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.multiview_tracking.schema import TrackingConfig
from src.tennis_scene.multiview_tracking.storage import track_store
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.multiview_tracking",
    fields=(
        BoundaryPathField(
            "candidates",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", required=True, type=Path, help="Frame pose candidate clip store"
    )
    parser.add_argument(
        "--candidate-artifact", help="Require this active frame pose artifact ID"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Separate output clip store"
    )
    parser.add_argument("--players-per-side", required=True, type=int, choices=(1, 2))
    args = parser.parse_args()
    paths = {
        "candidates": args.candidates.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=paths["output"],
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    resolved = PATH_BOUNDARY.validate(
        paths, resolver=PathResolver(roots), independent_artifact_inputs=True
    )
    ref = track_store(
        resolved.declared("candidates").path,
        resolved.declared("output").path,
        TrackingConfig(players_per_side=args.players_per_side),
        expected_artifact_id=args.candidate_artifact,
    )
    print(args.output.expanduser() / ref.path)


if __name__ == "__main__":
    main()
