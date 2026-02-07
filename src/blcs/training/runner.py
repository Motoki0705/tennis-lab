"""BLCS training runners using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.blcs.data.datamodule import BLCSDataModule, BLCSMultiViewDataModule
from src.blcs.training.lightning_module import BLCSLightningModule
from src.blcs.training.multiview_lightning_module import BLCSMultiViewLightningModule

_RUNNER_REGISTRY: dict[str, type[BaseTrainingRunner]] = {}


def _register(name: str):  # noqa: ANN202
    """Decorator to register a runner class under *name*."""

    def _wrap(cls: type[BaseTrainingRunner]) -> type[BaseTrainingRunner]:
        _RUNNER_REGISTRY[name] = cls
        return cls

    return _wrap


def select_runner(config: Any) -> BaseTrainingRunner:
    """Return the appropriate BLCS training runner for *config*.

    Selection is based on ``config.data.mode``:

    * ``"multiview"`` → :class:`BLCSMultiViewTrainingRunner`
    * anything else (``"default"``) → :class:`BLCSTrainingRunner`
    """
    mode = str(getattr(config.data, "mode", "default"))
    if mode in _RUNNER_REGISTRY:
        return _RUNNER_REGISTRY[mode]()
    if mode.startswith("multiview"):
        return BLCSMultiViewTrainingRunner()
    return BLCSTrainingRunner()


@_register("default")
class BLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for BLCS single-view model."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build BLCS data module."""
        return BLCSDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build BLCS lightning module."""
        return BLCSLightningModule(config)

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Log model parameters after dry run batch loading."""
        # Model parameter logging is handled in build_lightning_module
        pass


@_register("multiview")
class BLCSMultiViewTrainingRunner(BaseTrainingRunner):
    """Training runner for BLCS multi-view model."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build BLCS multi-view data module."""
        return BLCSMultiViewDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build BLCS multi-view lightning module."""
        return BLCSMultiViewLightningModule(config)
