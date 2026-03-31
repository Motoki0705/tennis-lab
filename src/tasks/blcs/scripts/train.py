"""Train a BLCS model with Hydra-managed configuration.

Usage:
    python -m src.tasks.blcs.scripts.train
    python -m src.tasks.blcs.scripts.train training.max_epochs=5 run.gpus=0
    python -m src.tasks.blcs.scripts.train model=multiview data=multiview
    python -m src.tasks.blcs.scripts.train run.dry_run=true

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/train.yaml`.
    - The runner handles the full BLCS training loop from the resolved config.
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
