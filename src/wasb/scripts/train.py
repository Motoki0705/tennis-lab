"""Train a WASB model with Hydra-managed configuration.

This is the **single** Hydra entry point for WASB training.
Use ``--config-name`` to select the training configuration:

    # ball detection (default)
    uv run python -m src.wasb.scripts.train

    # other variants (when added)
    uv run python -m src.wasb.scripts.train --config-name train_<variant>

Config entry point: ``src/wasb/configs/train_ball_detection.yaml`` (default)
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.wasb.training.runner import WASBTrainingRunner


def run_training(config: DictConfig) -> None:
    """Execute WASB training with the provided configuration."""
    runner = WASBTrainingRunner()
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train_ball_detection", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for WASB training."""
    run_training(config)


if __name__ == "__main__":
    main()
