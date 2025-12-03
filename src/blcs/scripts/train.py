"""Training script for BLCS.

Usage:
    python -m blcs.scripts.train [--config CONFIG_PATH] [--gpus NUM_GPUS]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.blcs.data.datamodule import BLCSDataModule
from src.blcs.training.lightning_module import BLCSLightningModule
from src.blcs.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BLCS model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/blcs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config or 'default'}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data module
    data_module = BLCSDataModule(config)

    # Initialize model
    model = BLCSLightningModule(config)
    logger.info(f"Model parameters: {model.model.get_num_params():,}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="blcs-{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=20,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )

    # Trainer
    train_cfg = config.get("training", {})
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 200),
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        precision="16-mixed" if args.gpus > 0 else 32,
        log_every_n_steps=50,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume,
    )

    # Test
    logger.info("Running test evaluation...")
    trainer.test(model, data_module)

    logger.info(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
