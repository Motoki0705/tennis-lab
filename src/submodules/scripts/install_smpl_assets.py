"""Install byte-verified SMPL auxiliary files from a user-acquired GVHMR checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES, install_smpl_assets
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
    name="submodules.install_smpl_assets",
    fields=(
        BoundaryPathField(
            "gvhmr_root", PathRole.EXTERNAL_ASSET, PathDirection.INPUT, PathKind.DIRECTORY,
            required=False, must_exist=True, allow_role_root=True,
        ),
        BoundaryPathField(
            "checkpoint_root", PathRole.CHECKPOINT, PathDirection.OUTPUT, PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-root", type=Path, help="Absolute official GVHMR checkout directory")
    parser.add_argument(
        "--checkpoint-root", type=Path, required=True,
        help="Absolute checkpoint root (normally $PWD/ckpt)",
    )
    parser.add_argument("--list", action="store_true", help="List sources, destinations and SHA-256")
    args = parser.parse_args()
    if not args.list and args.gvhmr_root is None:
        parser.error("--gvhmr-root is required unless --list is specified")
    if not args.checkpoint_root.is_absolute() or (
        args.gvhmr_root is not None and not args.gvhmr_root.is_absolute()
    ):
        parser.error("gvhmr-root and checkpoint-root must be explicit absolute paths")
    checkpoint_root = args.checkpoint_root.resolve()
    upstream = args.gvhmr_root.resolve() if args.gvhmr_root is not None else checkpoint_root
    roots = RuntimePathRoots(
        project_root=upstream, data_root=upstream, external_asset_root=upstream,
        checkpoint_root=checkpoint_root, output_root=checkpoint_root,
        artifact_root=checkpoint_root, cache_root=checkpoint_root,
    )
    values = {"checkpoint_root": args.checkpoint_root}
    if args.gvhmr_root is not None:
        values["gvhmr_root"] = args.gvhmr_root
    paths = PATH_BOUNDARY.validate(values, resolver=PathResolver(roots))
    checkpoint_root = paths.declared("checkpoint_root").path
    if args.list:
        for source in SMPL_ASSET_SOURCES.values():
            print(f"{source.sha256}  {checkpoint_root / source.checkpoint_path}")
            print(f"  {source.source_url}")
        return
    for path in install_smpl_assets(
        gvhmr_root=paths.declared("gvhmr_root").path, checkpoint_root=checkpoint_root,
    ):
        print(path)


if __name__ == "__main__":
    main()
