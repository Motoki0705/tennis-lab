"""Train a BLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.blcs.scripts.train`
    `uv run python -m src.blcs.scripts.train training.max_epochs=5 run.gpus=0`
    `uv run python -m src.blcs.scripts.train run.dry_run=true`

Config entry point: `src/blcs/configs/train.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

import logging
import os
import types
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.blcs.data.datamodule import BLCSDataModule
from src.blcs.training.lightning_module import BLCSLightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _select_devices(gpus: int) -> tuple[str, int]:
    """Return accelerator/devices based on requested GPU count."""
    if gpus > 0 and torch.cuda.is_available():
        return "gpu", gpus
    return "cpu", 1


def run_dry_run(config: DictConfig, output_dir: Path) -> None:
    """Verify config and dataloader by loading a single batch.

    Forces CPU mode to avoid CUDA init failures in restricted environments.
    """
    print("Running dry run (no training)...")
    # Force CPU-only to avoid CUDA init failures in restricted environments.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_args, **_kwargs: False, torch.cuda)
    torch.cuda.device_count = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)
    torch.cuda.current_device = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)

    # Initialize data module
    data_module = BLCSDataModule(config)
    data_module.num_workers = 0  # Avoid multiprocessing in restricted environments
    data_module.pin_memory = False
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    # Print batch shapes
    if isinstance(batch, dict):
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"  {key}: {tuple(value.shape)}")
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        if hasattr(inputs, "shape"):
            print(f"Loaded batch: inputs {tuple(inputs.shape)}")
        if hasattr(targets, "shape"):
            print(f"  targets {tuple(targets.shape)}")

    # Build model and run minimal trainer
    model = BLCSLightningModule(config)
    logger.info(f"Model parameters: {model.model.get_num_params():,}")

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, data_module)
    print(f"Dry run complete. Outputs saved to {output_dir}")


def run_training(config: DictConfig) -> None:
    """Run a single PL training job with the provided config."""
    pl.seed_everything(int(config.run.seed))

    output_dir = Path(to_absolute_path(str(config.run.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, output_dir / "config.yaml")

    # Check for dry run
    if getattr(config.run, "dry_run", False):
        run_dry_run(config, output_dir)
        return

    # Initialize data module
    data_module = BLCSDataModule(config)

    # Initialize model
    model = BLCSLightningModule(config)
    logger.info(f"Model parameters: {model.model.get_num_params():,}")

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )

    # Store checkpoints under the versioned log directory (like WASB)
    checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
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

    # Trainer
    train_cfg = config.get("training", {}) or {}
    accelerator, devices = _select_devices(int(config.run.gpus))
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 200),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        fast_dev_run=bool(config.run.fast_dev_run),
        precision="16-mixed" if accelerator == "gpu" else 32,
        log_every_n_steps=50,
        deterministic=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(
        model,
        data_module,
        ckpt_path=to_absolute_path(config.run.resume) if config.run.resume else None,
    )

    # Test
    if not bool(config.run.fast_dev_run):
        logger.info("Running test evaluation...")
        trainer.test(model, data_module)

    logger.info(f"Training complete. Outputs saved to {output_dir}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    run_training(config)


if __name__ == "__main__":
    main()
F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)
