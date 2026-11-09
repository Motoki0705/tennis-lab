"""Stabilize saved head-camera rotations while retaining positions and gaps (CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.visualization.player_view.stabilization import (
    StabilizationConfig,
    export_stabilized_cameras,
)
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
    name="tennis_scene.player_view_camera_stabilization",
    fields=(
        BoundaryPathField(
            "cameras",
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
    parser.add_argument("--cameras", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sigma-seconds", required=True, type=float)
    parser.add_argument("--truncate-sigma", default=3.0, type=float)
    args = parser.parse_args()
    config = StabilizationConfig(args.sigma_seconds, args.truncate_sigma)
    source, output = (
        args.cameras.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    roots = RuntimePathRoots(
        project_root=source.parent,
        data_root=source.parent,
        checkpoint_root=source.parent,
        artifact_root=source.parent,
        output_root=output,
        cache_root=output,
        external_asset_root=source.parent,
    )
    paths = PATH_BOUNDARY.validate(
        {"cameras": source, "output": output}, resolver=PathResolver(roots)
    )
    print(
        export_stabilized_cameras(
            cameras_dir=paths.declared("cameras").path,
            output_dir=paths.declared("output").path,
            config=config,
        )
    )


if __name__ == "__main__":
    main()
