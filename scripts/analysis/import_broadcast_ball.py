"""Curated venue-group import; copies only original media and quality-filtered 2D balls."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tennis_scene.dataset_pipeline.legacy import copy_legacy_broadcast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/tennis_multivew/processed/tennis_clip_source/dataset"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "data/tennis_multivew/processed/tennis_clip_source/curated_ball_v1"
        ),
    )
    args = parser.parse_args()
    groups = [
        "hamburg",
        "indoor_hard",
        "eastbourne",
        "monte_carlo",
        "shanghai",
        "washington",
        "shanghai",
        "hamburg",
        "outdoor_hard",
    ]
    copy_legacy_broadcast(
        args.source,
        args.destination,
        dataset_id="broadcast_curated_v1",
        video_groups={
            f"tennis_clip_source/clip_{i:03d}": f"broadcast_{group}"
            for i, group in enumerate(groups)
        },
        isolated_jump=0.12,
        neighbor_distance=0.06,
    )


if __name__ == "__main__":
    main()
