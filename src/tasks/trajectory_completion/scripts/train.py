"""Train a UV trajectory completion model using Hydra-managed configuration.

Example commands:
    `uv run python -m src.trajectory_completion.scripts.train`
    `uv run python -m src.trajectory_completion.scripts.train run.dry_run=true run.gpus=0 data.batch_size=2`

Config entry point: `src/trajectory_completion/configs/train.yaml`
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.trajectory_completion.training.runner import TrajectoryCompletionTrainingRunner


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    runner = TrajectoryCompletionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
