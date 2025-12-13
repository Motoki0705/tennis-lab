"""Training script for PLCS models using Hydra configurations."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.plcs.configs import PLCSConfig, register_configs
from src.plcs.data.datamodule import PLCSDataModule, PLCSSequenceDataModule
from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.training.sequence_lightning_module import PLCSSequenceLightningModule

register_configs()


def _select_devices(gpus: int) -> tuple[str, int]:
    """Return accelerator/devices tuple based on requested GPUs."""

    if gpus > 0:
        return "gpu", gpus
    return "cpu", 1


def _ensure_absolute(path: str | None) -> str | None:
    """Convert relative paths to absolute paths using the original CWD."""

    if path is None:
        return None
    return to_absolute_path(path)


def run_training(config: PLCSConfig) -> None:
    """Execute PLCS training with the provided configuration."""

    pl.seed_everything(config.training.seed)

    output_dir = Path(to_absolute_path(config.training.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    OmegaConf.save(config, output_dir / "config.yaml")

    is_sequence = config.data.mode == "sequence"
    if is_sequence:
        data_module = PLCSSequenceDataModule(config)
        model = PLCSSequenceLightningModule(config)
        checkpoint_prefix = "plcs-seq"
    else:
        data_module = PLCSDataModule(config)
        model = PLCSLightningModule(config)
        checkpoint_prefix = "plcs"

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
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

    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
    )

    accelerator, devices = _select_devices(config.training.gpus)

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        fast_dev_run=config.training.fast_dev_run,
        deterministic=True,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=_ensure_absolute(config.training.resume),
    )

    if not config.training.fast_dev_run:
        trainer.test(model, datamodule=data_module)

    print(f"Training complete. Outputs saved to {output_dir}")


@hydra.main(version_base=None, config_name="plcs")
def main(config: PLCSConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for frame-based PLCS training."""

    run_training(config)


if __name__ == "__main__":
    main()
