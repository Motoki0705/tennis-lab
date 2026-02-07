"""Train a BLCS model with Hydra-managed configuration.

This is the **single** Hydra entry point for all BLCS training variants.
Use ``--config-name`` to select the training configuration:

    # single-view (default)
    uv run python -m src.blcs.scripts.train

    # multiview
    uv run python -m src.blcs.scripts.train --config-name train_multiview

Config entry point: ``src/blcs/configs/train.yaml`` (default)
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.blcs.training.runner import select_runner


def run_training(config: DictConfig) -> None:
    """Execute BLCS training with the provided configuration."""
    runner = select_runner(config)
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    run_training(config)


if __name__ == "__main__":
    main()
