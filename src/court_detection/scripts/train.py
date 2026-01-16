"""Train court keypoint detection model.

Example:
    uv run python -m src.court_detection.scripts.train

    # With custom config
    uv run python -m src.court_detection.scripts.train model=hrnet_heatmap training.max_epochs=200

Config entry point: `src/court_detection/configs/train.yaml`
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.court_detection.data.datamodule import CourtKeypointDataModule
from src.court_detection.training.lightning_module import CourtKeypointLightningModule

LOGGER = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    """Train court keypoint detection model."""
    LOGGER.info("Starting training")
    LOGGER.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed
    pl.seed_everything(cfg.run.get("seed", 42))

    # Build datamodule
    datamodule = CourtKeypointDataModule(
        data_dir=cfg.data.get("data_dir", "data/court_detection/scenes"),
        batch_size=cfg.data.get("batch_size", 32),
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=cfg.data.get("pin_memory", True),
        input_size=tuple(cfg.data.get("input_size", [256, 256])),
        heatmap_size=tuple(cfg.data.get("heatmap_size", [64, 64])),
        augmentation=OmegaConf.to_container(cfg.data.get("augmentation", {})),
    )

    # Build model config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)

    # Build training config
    training_config = OmegaConf.to_container(cfg.training, resolve=True)

    # Build loss config
    loss_config = OmegaConf.to_container(cfg.loss, resolve=True)

    # Build lightning module
    lightning_module = CourtKeypointLightningModule(
        model_config=model_config,
        training_config=training_config,
        loss_config=loss_config,
    )

    # Build callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_config = cfg.training.get("checkpoint", {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.run.output_dir) / "checkpoints",
        filename="epoch_{epoch:03d}_pck_{val/pck:.4f}",
        save_top_k=checkpoint_config.get("save_top_k", 3),
        monitor=checkpoint_config.get("monitor", "val/pck"),
        mode=checkpoint_config.get("mode", "max"),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping_config = cfg.training.get("early_stopping", {})
    if early_stopping_config.get("patience", 0) > 0:
        early_stopping_callback = EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val/loss"),
            patience=early_stopping_config.get("patience", 20),
            mode=early_stopping_config.get("mode", "min"),
        )
        callbacks.append(early_stopping_callback)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Build logger
    logger = TensorBoardLogger(
        save_dir=cfg.run.output_dir,
        name="logs",
    )

    # Build trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.get("max_epochs", 100),
        accelerator="auto",
        devices=cfg.run.get("gpus", 1),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    # Train
    trainer.fit(lightning_module, datamodule=datamodule)

    # Test
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")

    LOGGER.info("Training complete")
    LOGGER.info("Best checkpoint: %s", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
