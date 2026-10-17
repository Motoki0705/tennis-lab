"""Use the existing DINO runner with fixed validation and checkpoint provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.annotation.artifacts import verify_reference
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.player_detection.training.runner import PlayerDetectionTrainingRunner
from src.utils.artifact_store import ArtifactStore

from .checkpoints import normalized_configuration
from .contracts import RunBinding
from .inputs import Inputs
from .precision import apply_precision


class BindingCallback(pl.Callback):
    def __init__(self, binding: RunBinding) -> None:
        self.binding = binding

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del trainer, pl_module
        checkpoint["annotation_learning"] = self.binding.model_dump(mode="json")

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del trainer, pl_module
        if checkpoint.get("annotation_learning") != self.binding.model_dump(
            mode="json"
        ):
            raise ValueError("player checkpoint cannot resume a different bound run")


class AnnotationPlayerRunner(PlayerDetectionTrainingRunner):
    def __init__(
        self, layout: AnnotationLayout, inputs: Inputs, binding: RunBinding
    ) -> None:
        self.layout, self.inputs, self.binding = layout, inputs, binding

    def prepare_config(self, config: Any) -> None:
        super().prepare_config(config)
        original = json.loads(
            verify_reference(
                self.layout, self.binding.initial_configuration
            ).read_bytes()
        )
        if normalized_configuration(config) != normalized_configuration(original):
            raise ValueError("player run configuration changed before training")

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return self.inputs.datamodule(self.layout, config)

    def apply_runtime_settings(self, config: TrainingRuntimeConfig) -> None:
        apply_precision(config.training)
        torch.backends.cudnn.benchmark = config.training.trainer.benchmark

    def build_trainer(
        self,
        config: Any,
        callbacks: list[Any],
        logger: TensorBoardLogger,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> pl.Trainer:
        trainer = super().build_trainer(
            config, callbacks, logger, artifact_store=artifact_store
        )
        # A valid mixed validation set may start with only confirmed absence.
        # mAP needs the full pinned support, not Lightning's first two batches.
        trainer.num_sanity_val_steps = 0
        return trainer

    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        self.prepare_config(config)
        if str(output_dir) != self.binding.output:
            raise ValueError("player logger must remain in the bound run")
        # Stable version preserves the original checkpoint directory on resume.
        return TensorBoardLogger(
            save_dir=str(output_dir), name="logs", version="learning"
        )

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        return [
            *super().callbacks_extra(config, datamodule, logger),
            BindingCallback(self.binding),
        ]
