"""Training script for PLCS model.

Usage:
    python -m plcs.scripts.train --config plcs/configs/default.yaml
    python -m plcs.scripts.train --epochs 50 --batch-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.plcs.data.datamodule import PLCSDataModule
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.utils.config import get_default_config, load_config, merge_configs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(description="Train PLCS model")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Override common parameters
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers")

    # Training options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plcs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (0 for CPU)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run quick test with 1 batch",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OmegaConf:
    """Build configuration from defaults and arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        OmegaConf: Merged configuration.

    """
    # Start with defaults
    config = get_default_config()

    # Load config file if provided
    if args.config is not None:
        file_config = load_config(args.config)
        config = merge_configs(config, file_config)

    # Apply CLI overrides
    overrides = {}

    if args.epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.hidden_dim is not None:
        overrides.setdefault("model", {})["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        overrides.setdefault("model", {})["num_layers"] = args.num_layers

    if overrides:
        config = merge_configs(config, overrides)

    return config


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    # Build config
    config = build_config(args)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")

    # Create data module
    data_module = PLCSDataModule(config)

    # Create model
    model = PLCSLightningModule(config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="plcs-{epoch:02d}",
            monitor="val/epoch_position_error_m",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/epoch_position_error_m",
            patience=10,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.get("gradient_clip_val", 1.0),
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # Test
    if not args.fast_dev_run:
        trainer.test(model, datamodule=data_module)

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
