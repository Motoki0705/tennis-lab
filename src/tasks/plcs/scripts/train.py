"""Train a PLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.tasks.plcs.scripts.train`
    `uv run python -m src.tasks.plcs.scripts.train run.gpus=0 training.max_epochs=1`
    `uv run python -m src.tasks.plcs.scripts.train run.dry_run=true`

Config entry point: `src/tasks/plcs/configs/train.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.tasks.plcs.training.runner import PLCSTrainingRunner

F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


def run_training(config: DictConfig) -> None:
    """Execute PLCS training with the provided configuration."""
    runner = PLCSTrainingRunner()
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
