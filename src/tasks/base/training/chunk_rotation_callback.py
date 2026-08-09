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
        max_epochs = trainer.max_epochs
        if (
            max_epochs is not None
            and max_epochs > 0
            and trainer.current_epoch + 1 >= max_epochs
        ):
            return
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "on_train_epoch_end"):
            datamodule.on_train_epoch_end()
