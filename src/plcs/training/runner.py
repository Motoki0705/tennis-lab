"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.plcs.data.datamodule import PLCSDataModule
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.training.lightning_module_kp3d import PLCSKeypoint3DLightningModule


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS.

    Non-kp3d models use unified I/O with a single datamodule and lightning module.
    kp3d remains a separated path by design.
    """

    def _is_kp3d_model(self, config: Any) -> bool:
        return str(getattr(config.model, "name", "")) == "plcs_kp3d"

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return PLCSDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        if self._is_kp3d_model(config):
            return PLCSKeypoint3DLightningModule(config)
        return PLCSLightningModule(config)
