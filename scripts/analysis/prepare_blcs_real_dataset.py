"""Create recording-disjoint BLCS pseudo-label segments from audited 2D balls."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tennis_scene.dataset_pipeline.blcs_training import export_blcs_training
from src.tennis_scene.dataset_pipeline.geometry import TriangulationSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--synthetic-source", type=Path, required=True)
    parser.add_argument("--replay-count", type=int, default=128)
    parser.add_argument("--external-asset-root", required=True)
    args = parser.parse_args()
    with initialize_config_dir(
        config_dir=str(Path("src/tennis_scene/configs").resolve()), version_base="1.3"
    ):
        cfg = compose(
            config_name="build_slcs_dataset",
            overrides=[f"paths.external_asset_root={args.external_asset_root}"],
        )
    export_blcs_training(
        cfg,
        args.destination,
        synthetic_source=args.synthetic_source,
        replay_count=args.replay_count,
        video_splits={"video_000": "train", "video_001": "val", "video_002": "test"},
        settings=TriangulationSettings(20.0, 65.0, (15.0, 28.0), (-0.1, 12.0)),
    )


if __name__ == "__main__":
    main()
