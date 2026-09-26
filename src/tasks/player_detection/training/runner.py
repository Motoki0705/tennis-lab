"""Training runner for DINO player-detection fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.datamodule import PlayerDetectionDataModule
from src.tasks.player_detection.training.lightning_module import (
    PlayerDetectionLightningModule,
)


class PlayerDetectionTrainingRunner(BaseTrainingRunner):
    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        return PlayerTrainingConfig.from_config(config).shared

    def prepare_config(self, config: Any) -> None:
        PlayerTrainingConfig.from_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return PlayerDetectionDataModule(PlayerTrainingConfig.from_config(config).data)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        del datamodule
        module = PlayerDetectionLightningModule(config, PlayerTrainingConfig.from_config(config))
        module.steps_per_epoch = steps_per_epoch
        return module

    def test_after_fit(
        self,
        trainer: pl.Trainer,
        lightning_module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        callbacks: list[Any],
    ) -> None:
        """Test the validation-selected checkpoint, not the last epoch."""
        del callbacks
        with self.resume_checkpoint_load_env("best"):
            trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")

    def run_dry_run(self, config: Any, output_dir: Path) -> None:
        """Data-only dry run: upstream DINO denoising is CUDA-only."""
        datamodule = PlayerDetectionDataModule(PlayerTrainingConfig.from_config(config).data)
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        print(
            "Loaded batch: images",
            [tuple(image.shape) for image in batch.images],
            "boxes",
            [tuple(boxes.shape) for boxes in batch.boxes_cxcywh],
        )
        print(f"Dry run complete (no model built). Outputs saved to {output_dir}")
