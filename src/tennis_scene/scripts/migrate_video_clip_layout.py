"""Migrate one recording-based tennis dataset to dataset/video/clip layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.clip_studio.migration import (
    apply_video_clip_migration,
    plan_video_clip_migration,
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
    name="tennis_scene.video_clip_migration",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this flag, only print the validated plan.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=data_root.parent,
        data_root=data_root,
        checkpoint_root=data_root,
        artifact_root=data_root,
        output_root=data_root,
        cache_root=data_root,
        external_asset_root=data_root,
    )
    paths = PATH_BOUNDARY.validate(
        {"data_root": data_root}, resolver=PathResolver(roots)
    )
    plan = plan_video_clip_migration(
        paths.declared("data_root").path, args.dataset_id
    )
    for video in plan.videos:
        print(f"{video.raw_directory} -> {plan.raw_dataset_directory / video.video_id}")
    print(f"processed -> {plan.processed_directory}")
    if not args.apply:
        print("dry run: pass --apply to perform this migration")
        return 0
    apply_video_clip_migration(plan)
    print("migration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
