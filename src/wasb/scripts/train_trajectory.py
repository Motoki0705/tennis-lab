from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.wasb.data.trajectory_datamodule import TrajectoryDataModule
from src.wasb.training.trajectory_lightning_module import TrajectoryLightningModule
from src.wasb.utils.checkpoint import resolve_resume_ckpt_path
from src.wasb.utils.config import load_config, merge_configs


def resolve_model_name(config: DictConfig, config_path: str | Path) -> str:
    model_cfg = None
    if hasattr(config, "get"):
        model_cfg = config.get("model")
    if model_cfg is None:
        model_cfg = getattr(config, "model", None)

    name = None
    if model_cfg is not None:
        if hasattr(model_cfg, "get"):
            name = model_cfg.get("name")
        else:
            name = getattr(model_cfg, "name", None)

    if name is not None and str(name).strip() != "":
        return str(name).strip()

    return Path(config_path).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trajectory completer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parents[1] / "configs" / "trajectory.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/trajectory",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (0 for CPU)")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a quick test with a single batch",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and data, fetch one batch, then exit without training.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OmegaConf:
    default_config = load_config(args.config)
    config = default_config

    overrides: dict = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr

    if overrides:
        config = merge_configs(config, overrides)

    return config


def _setup_logging(config: OmegaConf) -> None:
    log_cfg = getattr(config, "logging", None)
    if log_cfg is None:
        return

    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = getattr(log_cfg, "fmt", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    datefmt = getattr(log_cfg, "datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def run_dry_run(config: OmegaConf, output_dir: Path) -> None:
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

    print(
        f"Loaded batch: xy_input {tuple(xy.shape)}, "
        f"targets {tuple(targets.shape)}"
    )

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


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    config = build_config(args)
    _setup_logging(config)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    model_name = resolve_model_name(config, args.config)
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    if args.dry_run:
        run_dry_run(config, output_dir)
        return

    resume_ckpt = resolve_resume_ckpt_path(
        args_resume=args.resume,
        config=config,
        output_dir=output_dir,
    )

    datamodule = TrajectoryDataModule(config)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)

    module = TrajectoryLightningModule(config, steps_per_epoch=steps_per_epoch)

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

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

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        precision=config.training.precision,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)

    if not args.fast_dev_run:
        trainer.test(module, datamodule=datamodule)

        last_ckpt = checkpoint_dir / "last.ckpt"
        vis_dir = Path(logger.log_dir) / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        if last_ckpt.exists():
            visualize_script = (
                Path(__file__).parents[0]
                / "visualize_trajectory.py"
            )
            cmd = [
                sys.executable,
                str(visualize_script),
                "--config",
                args.config,
                "--checkpoint",
                str(last_ckpt),
                "--split",
                "test",
                "--num-samples",
                "8",
                "--output-dir",
                str(vis_dir),
                "--gpus",
                str(args.gpus),
            ]
            logging.getLogger(__name__).info(
                "Running visualization: %s", " ".join(cmd)
            )
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                logging.getLogger(__name__).warning(
                    "Visualization script failed with return code %s", exc.returncode
                )
        else:
            logging.getLogger(__name__).warning(
                "last.ckpt not found at %s; skipping visualization", last_ckpt
            )

    print(f"Training complete. Checkpoints saved under {checkpoint_dir}")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
