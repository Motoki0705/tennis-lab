"""Train a 3D-trajectory event detection model using Hydra-managed configuration.

Example commands:
    `uv run python -m src.evnet_detection.scripts.train_3d`
    `uv run python -m src.evnet_detection.scripts.train_3d run.dry_run=true`
    `uv run python -m src.evnet_detection.scripts.train_3d data.scene_dir=data/blcs run.dry_run=false`

Config entry point: `src/evnet_detection/configs/train_3d.yaml`
"""

# mypy: disable-error-code=misc

from __future__ import annotations

import logging
import os
import types
from pathlib import Path

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

from src.evnet_detection.data.datamodule import EventDetectionDataModule
from src.evnet_detection.training.lightning_module import EventDetectionLightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _select_devices(gpus: int) -> tuple[str, int]:
    if gpus > 0 and torch.cuda.is_available():
        return "gpu", gpus
    return "cpu", 1


def _force_cpu() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_a, **_k: False, torch.cuda)
    torch.cuda.device_count = types.MethodType(lambda *_a, **_k: 0, torch.cuda)
    torch.cuda.current_device = types.MethodType(lambda *_a, **_k: 0, torch.cuda)


def _make_dry_run_config(cfg: DictConfig) -> DictConfig:
    overrides = {
        "run": {"dry_run": True},
        "data": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "allow_dummy": True,
        },
    }
    return OmegaConf.merge(cfg, overrides)


def run_dry_run(cfg: DictConfig, output_dir: Path) -> None:
    """Run a 1-batch fit to validate wiring."""
    _force_cpu()
    dry_cfg = _make_dry_run_config(cfg)
    data_module = EventDetectionDataModule(dry_cfg)
    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))
    if isinstance(batch, dict):
        for key, value in batch.items():
            if hasattr(value, "shape"):
                logger.info("batch %s: %s", key, tuple(value.shape))
    model = EventDetectionLightningModule(dry_cfg)
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
    logger.info("Dry run complete: %s", output_dir)


def run_training(cfg: DictConfig) -> None:
    """Run training or a dry run."""
    pl.seed_everything(int(cfg.run.seed))

    output_dir = Path(to_absolute_path(str(cfg.run.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    if bool(cfg.run.dry_run):
        run_dry_run(cfg, output_dir)
        return

    data_module = EventDetectionDataModule(cfg)
    model = EventDetectionLightningModule(cfg)

    train_cfg = cfg.get("training", {}) or {}
    accelerator, devices = _select_devices(int(cfg.run.gpus))
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="event-detection-{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=5,
            mode="min",
            min_delta=1.0e-3,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=int(train_cfg.get("max_epochs", 1)),
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        logger=tb_logger,
        callbacks=callbacks,
        precision="16-mixed" if accelerator == "gpu" else 32,
        log_every_n_steps=50,
        deterministic=True,
    )
    logger.info("Starting training...")
    trainer.fit(
        model,
        data_module,
        ckpt_path=to_absolute_path(cfg.run.resume) if cfg.run.resume else None,
    )
    logger.info("Training complete. Outputs saved to %s", output_dir)


@hydra.main(config_path="../configs", config_name="train_3d", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    run_training(cfg)


if __name__ == "__main__":
    main()
