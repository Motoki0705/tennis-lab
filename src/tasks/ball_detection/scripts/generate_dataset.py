"""Build the unified ball frame store from TrackNet, Meiji and chat-annotation balls.

Usage (from a worktree, point the roots at the main checkout)::

    .venv/bin/python -m src.tasks.ball_detection.scripts.generate_dataset \
        paths.output_root=/abs/tennis-lab/outputs paths.data_root=/abs/tennis-lab/data

Every frame of every clip is stored. The dataset version directory must not
exist; the build publishes it atomically.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src.tasks.ball_detection.generate_dataset.frame_store.builder import build_dataset
from src.tasks.ball_detection.generate_dataset.frame_store.config import (
    FrameStoreBuildConfig,
    validate_generate_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "ball_detection.generate_dataset"
register_boundary_validator(_BOUNDARY, validate_generate_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_dataset",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    destination = build_dataset(FrameStoreBuildConfig.from_config(cfg))
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Published {destination}")


if __name__ == "__main__":
    main()
