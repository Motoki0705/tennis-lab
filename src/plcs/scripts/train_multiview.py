"""Train a multi-view PLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.plcs.scripts.train_multiview`
    `uv run python -m src.plcs.scripts.train_multiview run.gpus=0 training.max_epochs=1`
    `uv run python -m src.plcs.scripts.train_multiview run.dry_run=true`

Config entry point: `src/plcs/configs/train_multiview.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.plcs.training.runner import PLCSMultiViewTrainingRunner

F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


def run_training(config: DictConfig) -> None:
    """Execute multi-view PLCS training with the provided configuration."""
    runner = PLCSMultiViewTrainingRunner()
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train_multiview", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for multi-view PLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
