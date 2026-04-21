"""Shared callback for rotating chunked training data at epoch boundaries."""

from __future__ import annotations

import pytorch_lightning as pl


class ChunkRotationCallback(pl.Callback):
    """Call ``datamodule.on_train_epoch_end`` when the datamodule supports it."""

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "on_train_epoch_end"):
            datamodule.on_train_epoch_end()