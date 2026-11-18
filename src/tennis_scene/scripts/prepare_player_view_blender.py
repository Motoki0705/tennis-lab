"""Prepare portable Blender inputs from saved player-view scenes and cameras."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.tennis_scene.visualization.player_view.blender import (
    PlayerViewRenderConfig,
    prepare_blender_shot,
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
    name="tennis_scene.player_view_blender",
    fields=(
        BoundaryPathField(
            "bundle",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
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
    for name in ("bundle", "cameras", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ball-radius-m", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    source, cameras, output = (
        p.expanduser().resolve() for p in (args.bundle, args.cameras, args.output)
    )
    input_root = Path(os.path.commonpath((source.parent, cameras.parent)))
    roots = RuntimePathRoots(
        project_root=input_root,
        data_root=input_root,
        checkpoint_root=input_root,
        artifact_root=input_root,
        output_root=output,
        cache_root=output,
        external_asset_root=input_root,
    )
    paths = PATH_BOUNDARY.validate(
        {"bundle": source, "cameras": cameras, "output": output},
        resolver=PathResolver(roots),
    )
    print(
        prepare_blender_shot(
            bundle_dir=paths.declared("bundle").path,
            cameras_dir=paths.declared("cameras").path,
            output_dir=paths.declared("output").path,
            config=PlayerViewRenderConfig(
                ball_radius_m=args.ball_radius_m, samples=args.samples
            ),
        )
    )


if __name__ == "__main__":
    main()
