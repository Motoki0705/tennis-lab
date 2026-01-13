"""Train MAE on tennis domain videos using Hydra-managed configuration.

Example commands:
    `uv run python -m src.mae.scripts.train`
    `uv run python -m src.mae.scripts.train model.hidden_dim=512 training.max_epochs=200`
    `uv run python -m src.mae.scripts.train data.max_resolution=448 model.use_moe=true`

Config entry point: `src/mae/configs/train.yaml`
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

from src.mae.data import MAEDataModule
from src.mae.training import MAELightningModule

log = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks.

    Args:
        cfg: Hydra configuration.

    Returns:
        List of callbacks.

    """
    callbacks = []

    # Model checkpoint
    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_cfg.get("dirpath", "checkpoints"),
        filename=checkpoint_cfg.get("filename", "mae-{epoch:03d}-{val/loss:.4f}"),
        monitor=checkpoint_cfg.get("monitor", "val/loss"),
        mode=checkpoint_cfg.get("mode", "min"),
        save_top_k=checkpoint_cfg.get("save_top_k", 3),
        save_last=checkpoint_cfg.get("save_last", True),
    )
    callbacks.append(checkpoint)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Early stopping (optional)
    if cfg.get("early_stopping", {}).get("enabled", False):
        early_stop = EarlyStopping(
            monitor=cfg.early_stopping.get("monitor", "val/loss"),
            patience=cfg.early_stopping.get("patience", 50),
            mode=cfg.early_stopping.get("mode", "min"),
        )
        callbacks.append(early_stop)

    return callbacks


def setup_logger(cfg: DictConfig) -> TensorBoardLogger:
    """Setup TensorBoard logger.

    Args:
        cfg: Hydra configuration.

    Returns:
        TensorBoard logger.

    """
    logger_cfg = cfg.get("logger", {})
    return TensorBoardLogger(
        save_dir=logger_cfg.get("save_dir", "logs"),
        name=logger_cfg.get("name", "mae"),
        version=logger_cfg.get("version", None),
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train MAE model.

    Args:
        cfg: Hydra configuration.

    """
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # Setup data module
    log.info("Setting up data module...")
    datamodule = MAEDataModule.from_config(cfg)

    # Setup model
    log.info("Setting up model...")
    model = MAELightningModule.from_config(cfg)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model has {num_params:,} trainable parameters")

    # Setup callbacks and logger
    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)

    # Setup trainer
    trainer_cfg = cfg.get("trainer", {})
    trainer = pl.Trainer(
        max_epochs=cfg.get("training", {}).get("max_epochs", 400),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "16-mixed"),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", True),
        deterministic=trainer_cfg.get("deterministic", False),
    )

    # Train
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Save final model
    if trainer.is_global_zero:
        final_path = Path("checkpoints") / "mae-final.ckpt"
        trainer.save_checkpoint(final_path)
        log.info(f"Saved final checkpoint to {final_path}")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
