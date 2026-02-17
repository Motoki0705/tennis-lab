"""Callbacks for event-aware training behavior."""

from __future__ import annotations

import pytorch_lightning as pl


class EventBoostScheduleCallback(pl.Callback):
    """Linearly warm up event boost factor until configured max epoch."""

    def __init__(self, max_boost: float = 1.0, warmup_epochs: int = 10) -> None:
        self.max_boost = float(max_boost)
        self.warmup_epochs = max(int(warmup_epochs), 1)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not hasattr(pl_module, "event_boost"):
            return
        ratio = min(float(trainer.current_epoch + 1) / float(self.warmup_epochs), 1.0)
        pl_module.event_boost = self.max_boost * ratio
