"""BLCS training runners using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.blcs.data.datamodule import BLCSDataModule, BLCSMultiViewDataModule
from src.blcs.training.lightning_module import BLCSLightningModule
from src.blcs.training.multiview_lightning_module import BLCSMultiViewLightningModule


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

    def checkpoint_prefix(self, config: Any) -> str:
        """Return checkpoint filename prefix."""
        return "blcs"

    def checkpoint_monitor(self, config: Any) -> str:
        """Return metric to monitor for checkpointing."""
        return "val/loss"

    def early_stopping_monitor(self, config: Any) -> str:
        """Return metric to monitor for early stopping."""
        return "val/pos_error_m"

    def early_stopping_patience(self, config: Any) -> int:
        """Return early stopping patience."""
        return 5

    def early_stopping_min_delta(self, config: Any) -> float | None:
        """Return early stopping min_delta."""
        return 1.0e-3

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        """Return additional trainer kwargs."""
        kwargs: dict[str, Any] = {
            "log_every_n_steps": 50,
        }
        if accelerator == "gpu":
            kwargs["precision"] = "16-mixed"
        return kwargs

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Log model parameters after dry run batch loading."""
        # Model parameter logging is handled in build_lightning_module
        pass


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

    def checkpoint_prefix(self, config: Any) -> str:
        """Return checkpoint filename prefix."""
        return "blcs-multiview"

    def checkpoint_monitor(self, config: Any) -> str:
        """Return metric to monitor for checkpointing."""
        return "val/loss"

    def early_stopping_monitor(self, config: Any) -> str:
        """Return metric to monitor for early stopping."""
        return "val/loss"

    def early_stopping_patience(self, config: Any) -> int:
        """Return early stopping patience."""
        return 20

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        """Return additional trainer kwargs."""
        kwargs: dict[str, Any] = {
            "log_every_n_steps": 50,
        }
        if accelerator == "gpu":
            kwargs["precision"] = "16-mixed"
        return kwargs
