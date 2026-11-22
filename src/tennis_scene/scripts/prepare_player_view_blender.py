"""Prepare portable Blender inputs from saved player-view scenes and cameras."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.tennis_scene.visualization.player_view.blender import (
    PlayerViewRenderConfig,
    prepare_blender_shot,
)
from src.tennis_scene.visualization.player_view.lenses import EquidistantLensConfig
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
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--lens-profile", choices=["mission1-wide-approx"])
    parser.add_argument("--lens-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lens-render-scale", type=int)
    args = parser.parse_args()
    if args.lens_profile is None and (
        args.lens_size is not None or args.lens_render_scale is not None
    ):
        parser.error("Lens size and render scale require --lens-profile")
    lens = None
    if args.lens_profile is not None:
        width, height = (640, 360) if args.lens_size is None else args.lens_size
        lens = EquidistantLensConfig(
            width=width,
            height=height,
            render_scale=4
            if args.lens_render_scale is None
            else args.lens_render_scale,
        )
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
            frame_range=tuple(args.frames) if args.frames is not None else None,
            lens=lens,
        )
    )


if __name__ == "__main__":
    main()
