"""Train a UV trajectory completion model using Hydra-managed configuration.

Example commands:
    `uv run python -m src.trajectory_completion.scripts.train`
    `uv run python -m src.trajectory_completion.scripts.train run.dry_run=true run.gpus=0 data.batch_size=2`

Config entry point: `src/trajectory_completion/configs/train.yaml`
"""

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

from src.trajectory_completion.data.datamodule import TrajectoryCompletionDataModule
from src.trajectory_completion.training.lightning_module import (
    TrajectoryCompletionLightningModule,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _select_devices(gpus: int) -> tuple[str, int]:
    if gpus > 0 and torch.cuda.is_available():
        return "gpu", gpus
    return "cpu", 1


def _force_cpu() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_args, **_kwargs: False, torch.cuda)
    torch.cuda.device_count = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)
    torch.cuda.current_device = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)


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
    dm = TrajectoryCompletionDataModule(dry_cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            logger.info("batch %s: %s", k, tuple(v.shape))

    module = TrajectoryCompletionLightningModule(dry_cfg)
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
    trainer.fit(module, dm)
    logger.info("Dry run complete. Outputs: %s", output_dir)


def run_training(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.run.seed))

    output_dir = Path(to_absolute_path(str(cfg.run.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    if bool(cfg.run.dry_run):
        run_dry_run(cfg, output_dir)
        return

    dm = TrajectoryCompletionDataModule(cfg)
    module = TrajectoryCompletionLightningModule(cfg)

    train_cfg = cfg.get("training", {}) or {}
    accelerator, devices = _select_devices(int(cfg.run.gpus))
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="trajectory-completion-{epoch:02d}",
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
        max_epochs=int(train_cfg.get("max_epochs", 50)),
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        logger=tb_logger,
        callbacks=callbacks,
        fast_dev_run=bool(cfg.run.fast_dev_run),
        log_every_n_steps=50,
        deterministic=True,
        precision="16-mixed" if accelerator == "gpu" else 32,
    )
    logger.info("Starting training...")
    trainer.fit(module, dm, ckpt_path=to_absolute_path(cfg.run.resume) if cfg.run.resume else None)
    logger.info("Training complete. Outputs saved to %s", output_dir)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    run_training(cfg)


if __name__ == "__main__":
    main()
