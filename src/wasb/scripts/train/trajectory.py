"""Train the WASB trajectory completer (Hydra-based).

Example commands:
    `uv run python -m src.wasb.scripts.train.trajectory`
    `uv run python -m src.wasb.scripts.train.trajectory training.max_epochs=1 run.gpus=0`

Config entry point: `src/wasb/configs/train_trajectory.yaml`
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.wasb.data.trajectory_datamodule import TrajectoryDataModule
from src.wasb.training.trajectory_lightning_module import TrajectoryLightningModule
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


def run_dry_run(config: DictConfig, output_dir: Path) -> None:
    """Load a single batch and run a 1-step fit loop on CPU."""
    print("Running dry run (no training)...")
    import os
    import types

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_args, **_kwargs: False, torch.cuda)  # type: ignore[assignment]
    torch.cuda.device_count = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)  # type: ignore[assignment]
    torch.cuda.current_device = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)  # type: ignore[assignment]

    datamodule = TrajectoryDataModule(config)
    datamodule.num_workers = 0
    datamodule.pin_memory = False
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    xy = batch["xy_input_norm"]
    targets = batch["target_xy_norm"]

    print(f"Loaded batch: xy_input {tuple(xy.shape)}, targets {tuple(targets.shape)}")

    steps_per_epoch = len(train_loader)
    module = TrajectoryLightningModule(config, steps_per_epoch=steps_per_epoch)
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


def _maybe_run_visualization(
    *,
    last_ckpt: Path,
    vis_dir: Path,
    gpus: int,
    seed: int,
) -> None:
    if not last_ckpt.exists():
        logging.getLogger(__name__).warning(
            "last.ckpt not found at %s; skipping visualization", last_ckpt
        )
        return

    vis_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.wasb.scripts.visualize.trajectory",
        f"visualization.checkpoint={last_ckpt}",
        f"visualization.output_dir={vis_dir}",
        "visualization.split=test",
        "visualization.num_samples=8",
        f"run.gpus={gpus}",
        f"run.seed={seed}",
    ]
    logging.getLogger(__name__).info("Running visualization: %s", " ".join(map(str, cmd)))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logging.getLogger(__name__).warning(
            "Visualization script failed with return code %s", exc.returncode
        )


@hydra.main(config_path="../../configs", config_name="train_trajectory", version_base="1.3")
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

    datamodule = TrajectoryDataModule(config)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)

    module = TrajectoryLightningModule(config, steps_per_epoch=steps_per_epoch)

    logger = TensorBoardLogger(save_dir=str(output_dir), name="logs")

    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="trajectory-{epoch:02d}",
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
        _maybe_run_visualization(
            last_ckpt=checkpoint_dir / "last.ckpt",
            vis_dir=Path(logger.log_dir) / "vis",
            gpus=gpus,
            seed=seed,
        )

    print(f"Training complete. Checkpoints saved under {checkpoint_dir}")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
