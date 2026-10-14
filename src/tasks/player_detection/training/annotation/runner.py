"""Use the existing DINO runner with fixed validation and checkpoint provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.annotation.artifacts import verify_reference
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.training.runner import PlayerDetectionTrainingRunner

from .checkpoints import normalized_configuration
from .contracts import RunBinding
from .inputs import Inputs


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
