"""Train a WASB trajectory event detector (Hydra-based).

This trains a per-frame classifier over the `status` label in `Label.csv`:
    - 0: none
    - 1: shot
    - 2: bounce

Example commands:
    `uv run python -m src.wasb.scripts.train.event_detection`
    `uv run python -m src.wasb.scripts.train.event_detection run.dry_run=true`
    `uv run python -m src.wasb.scripts.train.event_detection training.max_epochs=1 run.gpus=0`

Config entry point: `src/wasb/configs/train_event_detection.yaml`
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.wasb.data.event_detection_datamodule import TrajectoryEventDataModule
from src.wasb.training.event_detection_lightning_module import (
    EventDetectionLightningModule,
)
from src.wasb.utils.checkpoint import resolve_resume_ckpt_path


def _setup_logging(config: DictConfig) -> None:
    log_cfg = getattr(config, "logging", None)
    if log_cfg is None:
        return

    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = getattr(log_cfg, "fmt", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    datefmt = getattr(log_cfg, "datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def _force_cpu_only() -> None:
    import os
    import types

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_args, **_kwargs: False, torch.cuda)
    torch.cuda.device_count = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)
    torch.cuda.current_device = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)


def run_dry_run(config: DictConfig, output_dir: Path) -> None:
    """Load a single batch and run a 1-step fit loop on CPU."""
    print("Running dry run (no training)...")
    _force_cpu_only()

    datamodule = TrajectoryEventDataModule(config)
    datamodule.num_workers = 0
    datamodule.pin_memory = False
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    xy = batch["xy_norm"]
    target = batch["target_status"]
    print(f"Loaded batch: xy_norm {tuple(xy.shape)}, target_status {tuple(target.shape)}")

    weights = None
    compute_cfg = getattr(config.training, "compute_class_weights", None)
    if compute_cfg is not None and bool(getattr(compute_cfg, "enabled", False)):
        counts = datamodule.estimate_class_counts(
            max_windows=int(getattr(compute_cfg, "max_windows", 200))
        )
        weights = datamodule.class_weights_from_counts(counts)
        bg_scale = float(getattr(config.training, "background_weight_scale", 1.0))
        if bg_scale != 1.0:
            weights = weights.clone()
            weights[0] = weights[0] * bg_scale

    steps_per_epoch = len(train_loader)
    module = EventDetectionLightningModule(
        config,
        steps_per_epoch=steps_per_epoch,
        class_weights=weights,
    )
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
    trainer.fit(module, datamodule=datamodule)
    (output_dir / "dry_run_ok.txt").write_text("ok\n", encoding="utf-8")


@hydra.main(config_path="../../configs", config_name="train_event_detection", version_base="1.3")  # type: ignore[misc]
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    seed = int(config.run.seed)
    pl.seed_everything(seed)

    _setup_logging(config)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    model_name = str(config.model.name)
    output_dir = Path(str(config.run.output_dir)) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    if bool(config.run.dry_run):
        run_dry_run(config, output_dir)
        return

    resume_ckpt = resolve_resume_ckpt_path(
        args_resume=None,
        config=config,
        output_dir=output_dir,
    )

    datamodule = TrajectoryEventDataModule(config)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)

    weights = None
    if config.training.class_weights is None:
        compute_cfg = getattr(config.training, "compute_class_weights", None)
        if compute_cfg is not None and bool(getattr(compute_cfg, "enabled", False)):
            counts = datamodule.estimate_class_counts(
                max_windows=int(getattr(compute_cfg, "max_windows", 5000))
            )
            weights = datamodule.class_weights_from_counts(counts)
            bg_scale = float(getattr(config.training, "background_weight_scale", 1.0))
            if bg_scale != 1.0:
                weights = weights.clone()
                weights[0] = weights[0] * bg_scale
            print(f"Estimated class counts: {counts.tolist()}")
            print(f"Using class weights: {weights.tolist()}")

    module = EventDetectionLightningModule(
        config,
        steps_per_epoch=steps_per_epoch,
        class_weights=weights,
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="logs")
    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="event-{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    gpus = int(config.run.gpus)
    trainer = pl.Trainer(
        max_epochs=int(config.training.max_epochs),
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=bool(config.run.fast_dev_run),
        precision=str(config.training.precision),
        deterministic=True,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)

    if not bool(config.run.fast_dev_run):
        trainer.test(module, datamodule=datamodule)

    print(f"Training complete. Checkpoints saved under {checkpoint_dir}")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
