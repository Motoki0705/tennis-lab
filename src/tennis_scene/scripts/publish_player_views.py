"""Validate Blender renders and publish labeled synchronized player-view videos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.tennis_scene.visualization.player_view.video import publish_player_views
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
    name="tennis_scene.player_view_videos",
    fields=(
        BoundaryPathField(
            "package",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "renders",
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
    for name in ("package", "renders", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Allow an explicitly selected preview of at most 12 frames",
    )
    args = parser.parse_args()
    package, renders, output = (
        p.expanduser().resolve() for p in (args.package, args.renders, args.output)
    )
    root = Path(os.path.commonpath((package.parent, renders.parent)))
    roots = RuntimePathRoots(
        project_root=root,
        data_root=root,
        checkpoint_root=root,
        artifact_root=root,
        output_root=output,
        cache_root=output,
        external_asset_root=root,
    )
    paths = PATH_BOUNDARY.validate(
        {"package": package, "renders": renders, "output": output},
        resolver=PathResolver(roots),
    )
    print(
        publish_player_views(
            package=paths.declared("package").path,
            renders=paths.declared("renders").path,
            output=paths.declared("output").path,
            preview=args.preview,
        )
    )


if __name__ == "__main__":
    main()
