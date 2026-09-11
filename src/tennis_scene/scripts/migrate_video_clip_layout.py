"""Migrate one recording-based tennis dataset to dataset/video/clip layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.clip_studio.migration import (
    apply_video_clip_migration,
    plan_video_clip_migration,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this flag, only print the validated plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = plan_video_clip_migration(args.data_root, args.dataset_id)
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
