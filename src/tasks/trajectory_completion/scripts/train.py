"""Train the trajectory completion model with Hydra-managed configuration.

Usage:
    python -m src.tasks.trajectory_completion.scripts.train
    python -m src.tasks.trajectory_completion.scripts.train run.dry_run=true run.gpus=0 data.batch_size=2

Notes:
    - Configuration is loaded from `src/tasks/trajectory_completion/configs/train.yaml`.
    - The script uses Hydra for configuration loading.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.trajectory_completion.training.runner import TrajectoryCompletionTrainingRunner


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    runner = TrajectoryCompletionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
