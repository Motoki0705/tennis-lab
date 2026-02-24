"""Train court keypoint detection model.

Example:
    uv run python -m src.tasks.court_detection.scripts.train

    # With custom config
    uv run python -m src.tasks.court_detection.scripts.train model=hrnet_heatmap training.max_epochs=200

Config entry point: `src/tasks/court_detection/configs/train.yaml`
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    """Train court keypoint detection model."""
    runner = CourtDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
