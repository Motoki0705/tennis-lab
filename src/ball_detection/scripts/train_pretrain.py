"""Train ball detection model on labeled data."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.ball_detection.training.runner import BallDetectionTrainingRunner


@hydra.main(config_path="../configs", config_name="train_pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    runner = BallDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
