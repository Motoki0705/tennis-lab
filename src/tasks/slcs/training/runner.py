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

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int:
        """Use the real window loader length for the step-based LR schedule."""
        if train_loader is None:
            datamodule.setup(stage="fit")
            train_loader = datamodule.train_dataloader()
        batches = len(train_loader)
        accumulate = int(config.training.trainer.accumulate_grad_batches)
        return max((batches + accumulate - 1) // accumulate, 1)


__all__ = ["SLCSTrainingRunner"]
