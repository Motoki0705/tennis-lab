"""Training script for WASB tennis models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torchvision.utils import save_image

from src.wasb.data.datamodule import TennisDataModule
from src.wasb.models import build_model
from src.wasb.training.lightning_module import WASBLightningModule
from src.wasb.utils.config import load_config, merge_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WASB tennis model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parents[1] / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/wasb",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument(
        "--backbone-lr", type=float, default=None, help="Backbone learning rate override"
    )
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

    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.backbone_lr is not None:
        overrides.setdefault("training", {})["backbone_learning_rate"] = args.backbone_lr

    if overrides:
        config = merge_configs(config, overrides)

    return config


def _setup_logging(config: OmegaConf) -> None:
    """Initialize Python logging from the config.

    Expects a ``logging`` section in the root config with keys:

        level: str   (e.g. "INFO", "DEBUG", ...)
        fmt: str     (logging format string)
        datefmt: str (date format string)
    """

    log_cfg = getattr(config, "logging", None)
    if log_cfg is None:
        return

    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = getattr(log_cfg, "fmt", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    datefmt = getattr(log_cfg, "datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def _save_sample_visuals(batch: dict, out_dir: Path) -> None:
    """Save example frame/heatmap pairs for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = batch["frames"][0]  # [T, C, H, W]
    heatmaps = batch["target_heatmaps"][0]  # [T_out, Hh, Wh]
    frame_paths = batch["frame_paths"]

    # Align target heatmaps with their corresponding frames (tail of the window).
    frame_start = max(len(frames) - len(heatmaps), 0)

    for idx, hm in enumerate(heatmaps):
        frame_idx = frame_start + idx
        img = frames[frame_idx] if frame_idx < len(frames) else frames[-1]

        frame_dst = out_dir / f"frame_{idx:02d}.png"
        save_image(img, frame_dst)

        hm_up = F.interpolate(
            hm.unsqueeze(0).unsqueeze(0),
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        hm_min, hm_max = hm_up.min(), hm_up.max()
        hm_norm = (hm_up - hm_min) / (hm_max - hm_min + 1e-6)
        heat_dst = out_dir / f"heatmap_{idx:02d}.png"
        save_image(hm_norm, heat_dst)

        overlay = torch.clamp(img + hm_norm.repeat(3, 1, 1) * 0.5, 0.0, 1.0)
        overlay_dst = out_dir / f"overlay_{idx:02d}.png"
        save_image(overlay, overlay_dst)

    # Recover the per-sample frame paths for the selected frames from the transposed collate.
    selected_paths: list[str] = []
    for idx in range(len(heatmaps)):
        frame_idx = frame_start + idx
        if frame_idx < len(frame_paths):
            # frame_paths is list-of-lists where outer index is frame position, inner is batch.
            selected_paths.append(frame_paths[frame_idx][0])
    if selected_paths:
        (out_dir / "frame_paths.txt").write_text("\n".join(selected_paths))


def run_dry_run(config: OmegaConf, output_dir: Path) -> None:
    """Verify config and dataloader by loading a single batch."""
    print("Running dry run (no training)...")
    # Force CPU-only to avoid CUDA init failures in restricted environments.
    import os
    import types

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_args, **_kwargs: False, torch.cuda)  # type: ignore[assignment]
    torch.cuda.device_count = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)  # type: ignore[assignment]
    torch.cuda.current_device = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)  # type: ignore[assignment]

    datamodule = TennisDataModule(config)
    datamodule.num_workers = 0  # Avoid multiprocessing in restricted environments
    datamodule.pin_memory = False
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    frames = batch["frames"]
    targets = batch["targets_norm"]
    target_heatmaps = batch["target_heatmaps"]
    visibility = batch["visibility"]

    print(
        f"Loaded batch: frames {tuple(frames.shape)}, "
        f"targets {tuple(targets.shape)}, "
        f"heatmaps {tuple(target_heatmaps.shape)}, "
        f"visibility {tuple(visibility.shape)}"
    )
    print(f"First sample frame paths: {batch['frame_paths'][0][:3]} ...")
    _save_sample_visuals(batch, output_dir / "dry_run")

    # Build model and lightning module
    steps_per_epoch = len(train_loader)
    model, io_handlers = build_model(config)
    backbone_ckpt = None
    model_cfg = config.get("model")
    if model_cfg and hasattr(model_cfg, "get"):
        backbone_ckpt = model_cfg.get("backbone_checkpoint")
    if backbone_ckpt and hasattr(model, "load_backbone_checkpoint"):
        model.load_backbone_checkpoint(backbone_ckpt)
    lightning_module = WASBLightningModule(
        config,
        model=model,
        steps_per_epoch=steps_per_epoch,
        io_handlers=io_handlers,
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

    trainer.fit(lightning_module, datamodule=datamodule)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    config = build_config(args)
    _setup_logging(config)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    if args.dry_run:
        run_dry_run(config, output_dir)
        return

    datamodule = TennisDataModule(config)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)

    model, io_handlers = build_model(config)
    backbone_ckpt = None
    model_cfg = config.get("model")
    if model_cfg and hasattr(model_cfg, "get"):
        backbone_ckpt = model_cfg.get("backbone_checkpoint")
    if backbone_ckpt and hasattr(model, "load_backbone_checkpoint"):
        model.load_backbone_checkpoint(backbone_ckpt)
    lightning_module = WASBLightningModule(
        config,
        model=model,
        steps_per_epoch=steps_per_epoch,
        io_handlers=io_handlers,
    )

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="wasb-{epoch:02d}",
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
        precision=config.training.precision
    )

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=args.resume)

    if not args.fast_dev_run:
        trainer.test(lightning_module, datamodule=datamodule)

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
