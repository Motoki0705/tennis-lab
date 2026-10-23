"""Court-only Trainer composition for a fixed number of optimizer updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.artifact_store import PublishingCheckpointIO
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.scaling.schedule import StepSchedule
from src.utils.artifact_store import ArtifactStore


def step_checkpoints(
    config: object, logger: TensorBoardLogger
) -> list[ModelCheckpoint]:
    schedule = StepSchedule.from_config(config)
    directory = Path(logger.log_dir) / "checkpoints"
    return [
        ModelCheckpoint(
            dirpath=directory,
            filename="update-{step:08d}",
            auto_insert_metric_name=False,
            monitor=None,
            save_top_k=-1,
            every_n_train_steps=schedule.checkpoint_every_n_steps,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=directory,
            filename="best-val-{step:08d}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
    ]


def build_step_trainer(
    config: object,
    callbacks: list[Any],
    logger: TensorBoardLogger,
    *,
    accelerator: str,
    devices: int,
    artifact_store: ArtifactStore | None,
) -> pl.Trainer:
    runtime = CourtTrainingConfig.from_config(config)
    trainer = runtime.shared.training.trainer
    schedule = StepSchedule.from_config(config)
    return pl.Trainer(
        max_epochs=-1,
        max_steps=schedule.max_steps,
        val_check_interval=schedule.validation_every_n_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        deterministic=trainer.deterministic,
        benchmark=trainer.benchmark,
        log_every_n_steps=trainer.log_every_n_steps,
        precision=trainer.precision,
        gradient_clip_val=trainer.gradient_clip_val,
        accumulate_grad_batches=trainer.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=trainer.reload_dataloaders_every_n_epochs,
        enable_progress_bar=trainer.enable_progress_bar,
        enable_model_summary=trainer.enable_model_summary,
        plugins=[PublishingCheckpointIO(artifact_store)]
        if artifact_store is not None and artifact_store.enabled
        else None,
    )
