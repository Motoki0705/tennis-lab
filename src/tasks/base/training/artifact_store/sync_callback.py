"""Periodic synchronization for non-checkpoint training artifacts."""

from __future__ import annotations

import time
from typing import Any

import pytorch_lightning as pl

from src.utils.artifact_store import ArtifactStore


class ArtifactSyncCallback(pl.Callback):
    """Mirror logs and task artifacts while training is running."""

    def __init__(self, store: ArtifactStore, *, interval_seconds: int) -> None:
        if interval_seconds <= 0:
            raise ValueError("artifact sync interval_seconds must be positive")
        self.store = store
        self.interval_seconds = interval_seconds
        self._last_sync = 0.0

    def _sync(self, trainer: pl.Trainer, *, force: bool) -> None:
        if not trainer.is_global_zero:
            return
        now = time.monotonic()
        if not force and now - self._last_sync < self.interval_seconds:
            return
        self.store.sync_tree()
        self._last_sync = now

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        self._sync(trainer, force=False)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del pl_module
        self._sync(trainer, force=True)

    def on_test_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del pl_module
        self._sync(trainer, force=True)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        self._sync(trainer, force=True)
