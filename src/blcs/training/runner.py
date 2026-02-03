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
