"""Train an event detection model using Hydra-managed configuration.

This is the **single** Hydra entry point for all event detection training
variants.  Use ``--config-name`` to select the training configuration:

    # UV-based (default)
    uv run python -m src.event_detection.scripts.train

    # 3D-trajectory
    uv run python -m src.event_detection.scripts.train --config-name train_3d

Config entry point: ``src/event_detection/configs/train_uv.yaml`` (default)
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.event_detection.training.runner import EventDetectionTrainingRunner


def run_training(config: DictConfig) -> None:
    """Execute event detection training with the provided configuration."""
    runner = EventDetectionTrainingRunner()
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train_uv", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for event detection training."""
    run_training(cfg)


if __name__ == "__main__":
    main()
