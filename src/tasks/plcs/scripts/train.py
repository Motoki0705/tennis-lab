"""Train a PLCS model with Hydra-managed configuration.

Usage:
    uv run python -m src.tasks.plcs.scripts.train
    uv run python -m src.tasks.plcs.scripts.train run.gpus=0 training.trainer.max_epochs=1
    uv run python -m src.tasks.plcs.scripts.train run.dry_run=true

Notes:
    - Configuration defaults are defined in YAML under `src/tasks/plcs/configs/`.
    - Temporary experiment changes should be provided via Hydra CLI overrides.
"""

# mypy: disable-error-code=misc

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.training.runner import PLCSTrainingRunner

F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


@dataclass
class PLCSRunConfig:
    output_dir: str
    seed: int | None
    gpus: int
    resume: str | None
    fast_dev_run: bool
    dry_run: bool


@dataclass
class PLCSTrainTrainerConfig:
    max_epochs: int
    gradient_clip_val: float | None
    deterministic: bool
    precision: str | None
    log_every_n_steps: int
    check_val_every_n_epoch: int
    benchmark: bool


@dataclass
class PLCSTrainCheckpointConfig:
    enabled: bool
    filename: str
    monitor: str
    mode: str
    save_top_k: int
    save_last: bool


@dataclass
class PLCSTrainEarlyStoppingConfig:
    enabled: bool
    monitor: str
    mode: str
    patience: int
    min_delta: float | None
    check_on_train_epoch_end: bool


@dataclass
class PLCSTrainLRMonitorConfig:
    enabled: bool
    interval: str


@dataclass
class PLCSTrainingConfig:
    trainer: PLCSTrainTrainerConfig
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    scheduler: str
    min_lr: float
    checkpoint: PLCSTrainCheckpointConfig
    early_stopping: PLCSTrainEarlyStoppingConfig
    lr_monitor: PLCSTrainLRMonitorConfig
    matmul_precision: str
    allow_tf32: bool


@dataclass
class PLCSTrainConfig:
    model: dict[str, Any]
    data: dict[str, Any]
    training: PLCSTrainingConfig
    loss: dict[str, Any]
    metrics: dict[str, Any]
    run: PLCSRunConfig
    hydra: dict[str, Any]


ConfigStore.instance().store(name="plcs_train_schema", node=PLCSTrainConfig)


def _validate_against_schema(config: DictConfig) -> DictConfig:
    schema = OmegaConf.structured(PLCSTrainConfig)
    return cast(DictConfig, OmegaConf.merge(schema, config))


def run_training(config: DictConfig) -> None:
    """Execute PLCS training with the provided configuration."""
    runner = PLCSTrainingRunner()
    runner.run(config)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    validated = _validate_against_schema(config)
    run_training(validated)


if __name__ == "__main__":
    main()
