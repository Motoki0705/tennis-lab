"""Generate per-player head cameras from a saved player-view scene bundle (CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

from .camera_bundle import export_head_cameras
from .head_camera import HeadCameraConfig

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.player_view_head_cameras",
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
            "output",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    defaults = HeadCameraConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument(
        "--horizontal-fov-degrees", type=float, default=defaults.horizontal_fov_degrees
    )
    parser.add_argument(
        "--face-offset-head-m", type=float, nargs=3, default=defaults.face_offset_head_m
    )
    parser.add_argument("--near-m", type=float, default=defaults.near_m)
    parser.add_argument("--far-m", type=float, default=defaults.far_m)
    args = parser.parse_args()
    config = HeadCameraConfig(
        width=args.width,
        height=args.height,
        horizontal_fov_degrees=args.horizontal_fov_degrees,
        face_offset_head_m=tuple(args.face_offset_head_m),
        near_m=args.near_m,
        far_m=args.far_m,
    )
    source = args.bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
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
        {"bundle": source, "output": output},
        resolver=PathResolver(roots),
    )
    print(
        export_head_cameras(
            bundle_dir=paths.declared("bundle").path,
            output_dir=paths.declared("output").path,
            config=config,
        )
    )


if __name__ == "__main__":
    main()
