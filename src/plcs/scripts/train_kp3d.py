"""Train PLCS keypoint-3D model with Hydra-managed configuration.

Example:
    `uv run python -m src.plcs.scripts.train_kp3d`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.plcs.training.runner import PLCSTrainingRunner

F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


def run_training(config: DictConfig) -> None:
    """Execute PLCS kp3d training with provided configuration."""
    runner = PLCSTrainingRunner()
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train_kp3d", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for kp3d training."""
    run_training(config)


if __name__ == "__main__":
    main()
