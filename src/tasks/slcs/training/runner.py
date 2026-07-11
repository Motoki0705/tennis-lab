"""SLCS training runner (thin BaseTrainingRunner specialization)."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.slcs.data.datamodule import SLCSDataModule
from src.tasks.slcs.training.lightning_module import SLCSLightningModule


class SLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for SLCS."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        backend = str(config.get("data", {}).get("backend", "default"))
        if backend != "default":
            raise ValueError(
                f"Unsupported data.backend={backend!r} for SLCS. Supported: ['default']."
            )
        return SLCSDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        module = SLCSLightningModule(config)
        if steps_per_epoch is not None:
            module.steps_per_epoch = steps_per_epoch  # type: ignore[assignment, unused-ignore]
        return module


__all__ = ["SLCSTrainingRunner"]
