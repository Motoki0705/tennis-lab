"""Training runner for ball_detection stages."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.ball_detection.data.datamodule import BallDetectionDataModule
from src.tasks.ball_detection.data.pseudo_datamodule import BallDetectionPseudoDataModule
from src.tasks.ball_detection.training.callbacks import EventBoostScheduleCallback
from src.tasks.ball_detection.training.lightning_module import BallDetectionLightningModule


class BallDetectionTrainingRunner(BaseTrainingRunner):
    """Runner supporting pretrain and self-train datamodule switching."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        mode = str(config.get("run", {}).get("mode", "pretrain"))
        if mode == "selftrain":
            return BallDetectionPseudoDataModule(config)
        return BallDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        _ = datamodule
        module = BallDetectionLightningModule(config)
        module.steps_per_epoch = steps_per_epoch
        return module

    def callbacks_extra(self, config: Any, datamodule: pl.LightningDataModule, logger) -> list[Any]:
        _ = datamodule
        _ = logger
        event_cfg = config.get("training", {}).get("event_schedule", {})
        if not bool(event_cfg.get("enabled", False)):
            return []
        return [
            EventBoostScheduleCallback(
                max_boost=float(event_cfg.get("max_boost", 1.0)),
                warmup_epochs=int(event_cfg.get("warmup_epochs", 10)),
            )
        ]
