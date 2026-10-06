"""Compose the refiner with the shared Lightning training lifecycle."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.ball_refiner_3d.configuration.training import (
    RefinerTrainingSettings,
    training_config,
)
from src.tasks.ball_refiner_3d.data.augmentation.audit import corruption_audit
from src.tasks.ball_refiner_3d.data.datamodule import RefinerDataModule
from src.tasks.ball_refiner_3d.evaluation.artifacts import evaluate_checkpoint
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.training.lightning_module import RefinerLightningModule
from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.utils.artifact_store import ArtifactStore
from src.utils.configuration import PathRole
from src.utils.io import write_json_atomic


class RefinerTrainingRunner(BaseTrainingRunner):
    def __init__(self) -> None:
        self._settings: RefinerTrainingSettings | None = None

    @property
    def settings(self) -> RefinerTrainingSettings:
        if self._settings is None:
            raise RuntimeError("Training configuration has not been validated")
        return self._settings

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        self._settings = training_config(config)
        return self._settings.runtime

    def prepare_config(self, config: Any) -> None:
        settings = self.settings
        if (
            settings.runtime.run.gpus > 0
            and not settings.runtime.run.dry_run
            and not os.environ.get("TENNIS_RUN_ID")
        ):
            raise RuntimeError(
                "GPU training must be launched through the shared training queue"
            )
        resume = settings.runtime.run.resume
        if resume is not None:
            if (
                resume.parent
                != settings.runtime.run.output_dir / "logs/version_0/checkpoints"
            ):
                raise ValueError(
                    "Resume continues the original output directory; use init_weights for a new run"
                )
            checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
            RefinerLightningModule.validate_resume_checkpoint(settings, checkpoint)
            manifest_hash = hashlib.sha256((settings.dataset / "manifest.json").read_bytes()).hexdigest()
            if checkpoint["manifest_sha256"] != manifest_hash:
                raise ValueError("Cannot resume against a different dataset manifest")
            if checkpoint["step"] >= settings.updates.steps:
                raise ValueError(
                    "The checkpoint has already completed this update budget"
                )

    def apply_runtime_settings(self, config: TrainingRuntimeConfig) -> None:
        super().apply_runtime_settings(config)
        torch.set_num_threads(self.settings.updates.cpu_threads)

    def build_datamodule(self, config: Any) -> RefinerDataModule:
        settings = self.settings
        module = RefinerDataModule(settings)
        module.setup("fit")
        evaluation = replace(
            settings.augmentation,
            event_probability=settings.raw["data"]["evaluation_event_probability"],
        )
        write_json_atomic(
            settings.runtime.run.output_dir / "data_contract.json",
            {
                "dataset": str(settings.dataset),
                "manifest_sha256": module.dataset.manifest_hash,
                "fps": module.dataset.fps,
                **{
                    f"{split}_ids": [r.name for r in module.dataset.split(split)]
                    for split in ("train", "val", "test")
                },
                "evaluation_corruption": asdict(evaluation),
                "evaluation_seed": settings.raw["data"]["evaluation_seed"],
                "validation_audit": corruption_audit(module.validation),
                "selection": "lowest all-frame physical-coordinate RMSE on fixed validation corruption",
                "augmentation_refresh": "training split refreshed every evaluate_every generator updates; seed+10000+block",
            },
        )
        return module

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int:
        return self.settings.updates.evaluate_every

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> RefinerLightningModule:
        return RefinerLightningModule(self.settings)

    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        path = self.settings.runtime.resolver.validate(PathRole.OUTPUT, output_dir)
        return TensorBoardLogger(save_dir=str(path), name="logs", version=0)

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        # Refiner schedules count generator updates, not Lightning's G+D optimizer steps.
        # The module drives the shared GAN strategy at that explicit boundary.
        return []

    def build_callbacks(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> list[Any]:
        callbacks = super().build_callbacks(
            config, datamodule, logger, artifact_store=artifact_store
        )
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.enable_version_counter = False
        return cast(list[Any], callbacks)

    def maybe_load_init_weights(
        self, config: TrainingRuntimeConfig, lightning_module: pl.LightningModule
    ) -> None:
        if config.run.init_weights is None:
            return
        if not isinstance(lightning_module, RefinerLightningModule):
            raise TypeError("Expected RefinerLightningModule")
        model, metadata = load_checkpoint(config.run.init_weights, torch.device("cpu"))
        if (
            model.config != self.settings.model
            or metadata["event_sigma_frames"] != self.settings.event.sigma_frames
        ):
            raise ValueError(
                "init_weights requires matching model and event-target contracts"
            )
        lightning_module.model.load_state_dict(model.state_dict(), strict=True)

    def resolve_resume(
        self, config: TrainingRuntimeConfig, output_dir: Path
    ) -> str | None:
        path = super().resolve_resume(config, output_dir)
        if path is not None:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("schema") != "ball_refiner_3d.events.v2"
                or "loops" not in checkpoint
            ):
                raise ValueError(
                    "Full-state resume requires a Lightning events.v2 checkpoint; use init_weights for legacy weights"
                )
        return cast(str | None, path)

    def test_after_fit(
        self,
        trainer: pl.Trainer,
        lightning_module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        callbacks: list[Any],
    ) -> None:
        if not isinstance(lightning_module, RefinerLightningModule) or not isinstance(
            datamodule, RefinerDataModule
        ):
            raise TypeError("Refiner training composition mismatch")
        if lightning_module.generator_updates != self.settings.updates.steps:
            raise RuntimeError(
                "Training ended before the configured generator update budget"
            )
        best = next(
            callback
            for callback in callbacks
            if isinstance(callback, ModelCheckpoint) and callback.monitor == "val/rmse"
        )
        last = next(
            callback
            for callback in callbacks
            if isinstance(callback, ModelCheckpoint) and callback.monitor is None
        )
        if not best.best_model_path or not last.best_model_path:
            raise RuntimeError(
                "Both validation-selected best and final checkpoints are required"
            )
        datamodule.setup("test")
        metadata = {
            "best_step": lightning_module.best_step,
            "training_seconds": time.monotonic() - lightning_module.start_time,
            "peak_gpu_memory_bytes": int(
                torch.cuda.max_memory_allocated(trainer.strategy.root_device)
            )
            if trainer.strategy.root_device.type == "cuda"
            else 0,
            "test_corruption_audit": corruption_audit(datamodule.test),
            "dataset_manifest_sha256": datamodule.dataset.manifest_hash,
        }
        output = self.settings.runtime.run.output_dir
        metrics = evaluate_checkpoint(
            Path(best.best_model_path),
            output / "predictions",
            datamodule.test,
            trainer.strategy.root_device,
            seed=self.settings.raw["data"]["evaluation_seed"],
            batch_size=self.settings.updates.batch_size,
            common_metadata=metadata,
        )
        final = evaluate_checkpoint(
            Path(last.best_model_path),
            output / "predictions_last",
            datamodule.test,
            trainer.strategy.root_device,
            seed=self.settings.raw["data"]["evaluation_seed"],
            batch_size=self.settings.updates.batch_size,
            common_metadata=metadata,
        )
        write_json_atomic(
            output / "state.json",
            {
                "status": "complete",
                "step": lightning_module.generator_updates,
                **metrics,
                "last_metrics": final,
            },
        )


def run_training(config: DictConfig) -> Path:
    runner = RefinerTrainingRunner()
    runner.run(config)
    return cast(Path, runner.settings.runtime.run.output_dir)
