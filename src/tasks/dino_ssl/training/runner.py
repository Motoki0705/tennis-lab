"""Training runner for DINOv3 tennis SSL."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.dino_ssl.data import build_dino_ssl_datamodule
from src.tasks.dino_ssl.training.lightning_module import DinoSSLLightningModule


class DinoSSLTrainingRunner(BaseTrainingRunner):
    """Self-supervised training runner (no labelled test phase)."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return build_dino_ssl_datamodule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return DinoSSLLightningModule(config, steps_per_epoch=steps_per_epoch)

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int | None:
        if train_loader is not None:
            return len(train_loader)
        datamodule.setup(stage="fit")
        return len(datamodule.train_dataloader())

    def skip_test(self, config: Any) -> bool:
        # Self-supervised pretraining has no labelled test set.
        return True


__all__ = ["DinoSSLTrainingRunner"]
