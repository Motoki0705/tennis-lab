"""Train the PLCS model with Hydra-managed configuration.

Usage:
    python -m src.tasks.plcs.scripts.train
    python -m src.tasks.plcs.scripts.train run.gpus=0 training.max_epochs=1
    python -m src.tasks.plcs.scripts.train training=gan_base
    python -m src.tasks.plcs.scripts.train run.dry_run=true

Notes:
    - Configuration is loaded from `src/tasks/plcs/configs/train.yaml`.
    - Experiment configs can be selected with `--config-name`.
    - GAN training is selected with a GAN training config.
    - The script uses Hydra for configuration loading.
    - Use `--config-name train_tracking` for multi-person tracking.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.utils.hydra import hydra_main


def run_training(config: DictConfig) -> None:
    """Execute PLCS training with the provided configuration."""
    runner = PLCSTrainingRunner()
    runner.run(config)


@hydra_main(
    config_path="../configs",
    config_name="train",
    version_base="1.3",
    validation_boundary="plcs.train",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    PLCSTrainingConfig.from_config(config)
    run_training(config)


if __name__ == "__main__":
    main()
