"""Lightning callback for rotating training chunks at epoch boundaries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl

if TYPE_CHECKING:
    from src.tasks.blcs.data.chunked_datamodule import ChunkedBLCSDataModule

logger = logging.getLogger(__name__)


class ChunkRotationCallback(pl.Callback):
    """Rotates the training chunk after ``epochs_per_chunk`` epochs.

    This callback calls :meth:`ChunkedBLCSDataModule.on_train_epoch_end` at
    the end of each training epoch, which internally handles the counter and
    chunk swap logic.
    """

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "on_train_epoch_end"):
            datamodule.on_train_epoch_end()
