"""Train a BLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.tasks.blcs.scripts.train`
    `uv run python -m src.tasks.blcs.scripts.train training.max_epochs=5 run.gpus=0`
    `uv run python -m src.tasks.blcs.scripts.train model=multiview data=multiview`
    `uv run python -m src.tasks.blcs.scripts.train run.dry_run=true`

Config entry point: `src/tasks/blcs/configs/train.yaml`
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.blcs.training.runner import BLCSTrainingRunner


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entry point."""
    runner = BLCSTrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()
