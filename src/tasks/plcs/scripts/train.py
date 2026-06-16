"""Train the PLCS model with Hydra-managed configuration.

Usage:
    python -m src.tasks.plcs.scripts.train
    python -m src.tasks.plcs.scripts.train run.gpus=0 training.max_epochs=1
    python -m src.tasks.plcs.scripts.train data=chunked_multiview_sequence_bs8
    python -m src.tasks.plcs.scripts.train --config-name train_chunked_gan
    python -m src.tasks.plcs.scripts.train run.dry_run=true

Notes:
    - Configuration is loaded from `src/tasks/plcs/configs/train.yaml`.
    - Experiment configs can be selected with `--config-name`.
    - Chunked training is selected with a chunked data config.
    - GAN training is selected with a GAN training config.
    - The script uses Hydra for configuration loading.
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


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
