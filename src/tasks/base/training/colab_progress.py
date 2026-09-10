"""Publish rank-zero Lightning progress for the Colab workflow monitor."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch


class ColabProgressCallback(pl.Callback):
    """Persist atomic observations at most once every five seconds during batches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.last_write = float("-inf")

    def _publish(
        self,
        trainer: pl.Trainer,
        phase: str,
        *,
        force: bool = False,
        batch_loss: Any = None,
    ) -> None:
        if not trainer.is_global_zero:
            return
        now = time.monotonic()
        if not force and now - self.last_write < 5:
            return
        metrics: dict[str, float] = {}
        observations: dict[str, Any] = dict(trainer.callback_metrics)
        if batch_loss is not None:
            observations["batch/loss"] = batch_loss
        non_finite: list[str] = []
        for name, value in observations.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            if isinstance(value, (int, float)):
                if math.isfinite(value):
                    metrics[str(name)] = float(value)
                else:
                    non_finite.append(str(name))
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "phase": phase,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "metrics": metrics,
            "non_finite_metrics": non_finite,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)
        self.last_write = now

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        self._publish(trainer, "training", batch_loss=loss)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._publish(
            trainer, "sanity_check" if trainer.sanity_checking else "validation"
        )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._publish(
            trainer,
            "sanity_check" if trainer.sanity_checking else "validation",
            force=True,
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._publish(trainer, "fit_finished", force=True)
