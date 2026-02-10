"""BLCS training runner using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.blcs.data.datamodule import BLCSDataModule, BLCSMultiViewDataModule
from src.blcs.training.lightning_module import BLCSLightningModule


class BLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for BLCS models.

    The concrete training path is selected from `config.data.output_mode`.
    """

    def _select_by_output_mode(
        self,
        config: Any,
        *,
        single_factory: Any,
        multiview_factory: Any,
        component_name: str,
    ) -> Any:
        """Select component by `config.data.output_mode`."""
        output_mode = str(config.data.output_mode)
        if output_mode == "single":
            return single_factory()
        elif output_mode == "multiview":
            return multiview_factory()
        else:
            raise ValueError(
                "Invalid config.data.output_mode="
                f"'{output_mode}' for {component_name}. "
                "Supported: ['single', 'multiview']"
            )

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build BLCS data module according to `config.data.output_mode`."""
        return self._select_by_output_mode(
            config,
            single_factory=lambda: BLCSDataModule(config),
            multiview_factory=lambda: BLCSMultiViewDataModule(config),
            component_name="datamodule",
        )

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build BLCS lightning module according to `config.data.output_mode`."""
        return BLCSLightningModule(config)

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Log model parameters after dry run batch loading."""
        # Model parameter logging is handled in build_lightning_module
        pass
