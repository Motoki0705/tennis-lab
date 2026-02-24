"""Train ball detection model on pseudo labels."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner


@hydra.main(config_path="../configs", config_name="train_selftrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.run.mode = "selftrain"
    print(OmegaConf.to_yaml(cfg))
    runner = BallDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
