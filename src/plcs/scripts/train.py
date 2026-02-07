"""Train a PLCS model with Hydra-managed configuration.

This is the **single** Hydra entry point for all PLCS training variants.
Use ``--config-name`` to select the training configuration:

    # frame-based (default)
    uv run python -m src.plcs.scripts.train

    # sequence-based
    uv run python -m src.plcs.scripts.train --config-name train_sequence

    # multiview
    uv run python -m src.plcs.scripts.train --config-name train_multiview

    # keypoint-3D
    uv run python -m src.plcs.scripts.train --config-name train_kp3d

Config entry point: ``src/plcs/configs/train.yaml`` (default)
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.plcs.training.runner import select_runner


def run_training(config: DictConfig) -> None:
    """Execute PLCS training with the provided configuration."""
    runner = select_runner(config)
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
