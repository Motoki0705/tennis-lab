"""Explicit family registration without constructing production-sized models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
)


@dataclass(frozen=True)
class ModelFamily:
    """Task-local factories; checkpoint lifecycle tests can reuse build/predict.

    The real supervised result includes the decoded prediction and real loss.
    No model, loss, batch or parameter defaults are supplied by the kit.
    """

    name: str
    compose: Callable[[Path, bool], DictConfig]
    build: Callable[[DictConfig], tuple[Any, Any]]
    batch: Callable[[DictConfig], Any]
    supervised_result: Callable[[Any, Any], Mapping[str, Any]]
    predict: Callable[[Any, Any], Any]
    save: Callable[[Any, Path], None]
    restore: Callable[[DictConfig, Path], Any]
    parameters: Callable[[Any], tuple[ParameterContract, tuple[OptimizerContract, ...]]]
    assert_batch: Callable[[Any], None]
    assert_prediction: Callable[[Any], None]
    # Task-owned adapter for non-Lightning loops or a task-specific GAN schedule.
    fit: Callable[..., Any] | None = None
    # Construct fresh modules/data without regenerating the immutable fixture.
    rebuild: Callable[[DictConfig], tuple[Any, Any]] | None = None
    training_state: Callable[[Any], Any] | None = None
    predict_artifact: Callable[[Path, Any], Any] | None = None


def save_checkpoint(trainer: pl.Trainer, path: Path) -> None:
    """Registered full training artifact boundary, reusable by lifecycle tests."""
    trainer.save_checkpoint(path)


def configured_selectors(config_dir: Path, *, identity: str = "name") -> dict[str, str]:
    selectors: dict[str, str] = {}
    with initialize_config_dir(
        config_dir=str(config_dir.resolve()), version_base="1.3"
    ):
        for path in sorted((config_dir / "model").glob("*.yaml")):
            if not path.name.startswith("_"):
                config = compose(config_name="train", overrides=[f"model={path.stem}"])
                selectors[path.stem] = str(config.model[identity])
    assert selectors, f"no public model selectors in {config_dir}"
    return selectors


def assert_selector_coverage(
    discovered: Mapping[str, str], registered: Mapping[str, str]
) -> None:
    assert discovered == registered, (
        f"model selector coverage: unregistered={sorted(discovered.keys() - registered.keys())}, "
        f"stale={sorted(registered.keys() - discovered.keys())}, "
        f"identity mismatch={sorted(key for key in discovered.keys() & registered.keys() if discovered[key] != registered[key])}"
    )
