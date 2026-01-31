"""Training runner for court keypoint detection."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.base.training.runner import BaseTrainingRunner
from src.court_detection.data.datamodule import CourtKeypointDataModule
from src.court_detection.training.lightning_module import CourtKeypointLightningModule


class CourtDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for court keypoint detection.

    Overrides datamodule/model construction and checkpoint monitoring for PCK.
    """

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build CourtKeypointDataModule from config."""
        return CourtKeypointDataModule(
            data_dir=config.data.get("data_dir", "data/court_detection/scenes"),
            batch_size=config.data.get("batch_size", 32),
            num_workers=config.data.get("num_workers", 4),
            pin_memory=config.data.get("pin_memory", True),
            input_size=tuple(config.data.get("input_size", [256, 256])),
            heatmap_size=tuple(config.data.get("heatmap_size", [64, 64])),
            augmentation=OmegaConf.to_container(config.data.get("augmentation", {})),
        )

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build CourtKeypointLightningModule from config."""
        model_config = OmegaConf.to_container(config.model, resolve=True)
        training_config = OmegaConf.to_container(config.training, resolve=True)
        loss_config = OmegaConf.to_container(config.loss, resolve=True)
        data_config = OmegaConf.to_container(config.data, resolve=True)

        return CourtKeypointLightningModule(
            model_config=model_config,
            training_config=training_config,
            loss_config=loss_config,
            data_config=data_config,
        )

    def checkpoint_monitor(self, config: Any) -> str:
        """Monitor PCK for checkpointing."""
        checkpoint_cfg = getattr(config.training, "checkpoint", {})
        return checkpoint_cfg.get("monitor", "val/pck")

    def checkpoint_mode(self, config: Any) -> str:
        """Higher PCK is better."""
        checkpoint_cfg = getattr(config.training, "checkpoint", {})
        return checkpoint_cfg.get("mode", "max")

    def checkpoint_prefix(self, config: Any) -> str:
        """Checkpoint filename prefix."""
        return "epoch_{epoch:03d}_pck_{val/pck:.4f}"

    def early_stopping_monitor(self, config: Any) -> str:
        """Monitor val/loss for early stopping."""
        early_cfg = getattr(config.training, "early_stopping", {})
        return early_cfg.get("monitor", "val/loss")

    def early_stopping_mode(self, config: Any) -> str:
        """Lower loss is better."""
        early_cfg = getattr(config.training, "early_stopping", {})
        return early_cfg.get("mode", "min")

    def early_stopping_patience(self, config: Any) -> int:
        """Early stopping patience."""
        early_cfg = getattr(config.training, "early_stopping", {})
        return early_cfg.get("patience", 20)

    def early_stopping_enabled(self, config: Any) -> bool:
        """Enable early stopping if patience > 0."""
        early_cfg = getattr(config.training, "early_stopping", {})
        return early_cfg.get("patience", 0) > 0

    def lr_monitor_interval(self, config: Any) -> str:
        """Log learning rate per epoch (matching original train.py)."""
        return "epoch"

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        """Additional trainer kwargs for court detection."""
        return {
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 1,
        }
