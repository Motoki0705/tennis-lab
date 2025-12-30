"""Train a PLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.plcs.scripts.train`
    `uv run python -m src.plcs.scripts.train run.gpus=0 training.max_epochs=1`
    `uv run python -m src.plcs.scripts.train run.dry_run=true`

Config entry point: `src/plcs/configs/train.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

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

from src.plcs.data.datamodule import PLCSDataModule, PLCSSequenceDataModule
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.training.sequence_lightning_module import PLCSSequenceLightningModule


def _select_devices(gpus: int) -> tuple[str, int]:
    """Return accelerator/devices tuple based on requested GPUs."""
    if gpus > 0:
        return "gpu", gpus
    return "cpu", 1


def _ensure_absolute(path: str | None) -> str | None:
    """Convert relative paths to absolute paths using the original CWD."""
    if path is None:
        return None
    return str(to_absolute_path(path))


F = TypeVar("F", bound=Callable[..., object])
hydra.main = cast(Callable[..., Callable[[F], F]], hydra.main)


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

    is_sequence = str(config.data.mode) == "sequence"
    if is_sequence:
        data_module = PLCSSequenceDataModule(config)
        model = PLCSSequenceLightningModule(config)
    else:
        data_module = PLCSDataModule(config)
        model = PLCSLightningModule(config)

    data_module.num_workers = 0  # Avoid multiprocessing in restricted environments
    data_module.pin_memory = False
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    # Print batch shapes
    if isinstance(batch, dict):
        print("Loaded batch:")
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"  {key}: {tuple(value.shape)}")
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        if hasattr(inputs, "shape"):
            print(f"Loaded batch: inputs {tuple(inputs.shape)}")
        if hasattr(targets, "shape"):
            print(f"  targets {tuple(targets.shape)}")

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

    trainer.fit(model, datamodule=data_module)
    print(f"Dry run complete. Outputs saved to {output_dir}")


def run_training(config: DictConfig) -> None:
    """Execute PLCS training with the provided configuration."""
    pl.seed_everything(int(config.run.seed))

    output_dir = Path(to_absolute_path(str(config.run.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    OmegaConf.save(config, output_dir / "config.yaml")

    # Check for dry run
    if getattr(config.run, "dry_run", False):
        run_dry_run(config, output_dir)
        return

    is_sequence = str(config.data.mode) == "sequence"
    if is_sequence:
        data_module = PLCSSequenceDataModule(config)
        model = PLCSSequenceLightningModule(config)
        checkpoint_prefix = "plcs-seq"
    else:
        data_module = PLCSDataModule(config)
        model = PLCSLightningModule(config)
        checkpoint_prefix = "plcs"

    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
    )

    # Store checkpoints under the versioned log directory (like WASB)
    checkpoint_dir = Path(logger.log_dir) / "checkpoints"

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{checkpoint_prefix}-{{epoch:02d}}",
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

    accelerator, devices = _select_devices(int(config.run.gpus))

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        fast_dev_run=bool(config.run.fast_dev_run),
        deterministic=True,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=_ensure_absolute(config.run.resume),
    )

    if not bool(config.run.fast_dev_run):
        trainer.test(model, datamodule=data_module)

    print(f"Training complete. Outputs saved to {output_dir}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for PLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
