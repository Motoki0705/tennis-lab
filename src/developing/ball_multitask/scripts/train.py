"""Train the ball multi-task model with Hydra configuration.

Example commands:
    `uv run python -m src.developing.ball_multitask.scripts.train`
    `uv run python -m src.developing.ball_multitask.scripts.train run.dry_run=true`
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.developing.ball_multitask.training.runner import BallMultitaskTrainingRunner


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    runner = BallMultitaskTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
