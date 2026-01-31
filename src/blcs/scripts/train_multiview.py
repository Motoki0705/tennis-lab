"""Train a multi-view BLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.blcs.scripts.train_multiview`
    `uv run python -m src.blcs.scripts.train_multiview training.max_epochs=5 run.gpus=0`
    `uv run python -m src.blcs.scripts.train_multiview run.dry_run=true`

Config entry point: `src/blcs/configs/train_multiview.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import hydra
from omegaconf import DictConfig

from src.blcs.training.runner import BLCSMultiViewTrainingRunner

F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


@hydra.main(config_path="../configs", config_name="train_multiview", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for multi-view BLCS training."""
    runner = BLCSMultiViewTrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()
