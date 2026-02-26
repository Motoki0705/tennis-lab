"""Train MAE on tennis domain videos using Hydra-managed configuration.

Example commands:
    `uv run python -m src.developing.mae.scripts.train`
    `uv run python -m src.developing.mae.scripts.train model.hidden_dim=512 training.max_epochs=200`
    `uv run python -m src.developing.mae.scripts.train data=cached_batches data.bucket_alpha=2.5`

Config entry point: `src/developing/mae/configs/train.yaml`
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.developing.mae.training.runner import MAETrainingRunner

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train MAE model.

    Args:
        cfg: Hydra configuration.

    """
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    runner = MAETrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    cast(Callable[[], None], main)()
