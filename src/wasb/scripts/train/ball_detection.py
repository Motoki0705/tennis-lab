"""Training script for WASB tennis models.

Run with Hydra-style overrides, for example:

```
uv run python -m src.wasb.scripts.train.ball_detection training.max_epochs=50 data.batch_size=32
```

Config entry point: `src/wasb/configs/train_ball_detection.yaml`
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
from torch.nn import functional as F
from torchvision.utils import save_image

from src.wasb.data.ball_detection_datamodule import BallDetectionDataModule
from src.wasb.data.patch_embeddings_datamodule import PatchEmbeddingsDataModule
from src.wasb.data.curriculum_sampling import (
    CurriculumStepCallback,
    VisibilityCurriculumSampler,
)
from src.wasb.models import build_model
from src.wasb.training import WASBLightningModule
from src.wasb.utils.checkpoint import resolve_resume_ckpt_path


def _setup_logging(config: DictConfig) -> None:
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


def run_dry_run(config: DictConfig, output_dir: Path) -> None:
    """Verify config and dataloader by loading a single batch."""
    print("Running dry run (no training)...")
    # Force CPU-only to avoid CUDA init failures in restricted environments.
    import os
    import types

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.cuda.is_available = types.MethodType(lambda *_args, **_kwargs: False, torch.cuda)  # type: ignore[assignment]
    torch.cuda.device_count = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)  # type: ignore[assignment]
    torch.cuda.current_device = types.MethodType(lambda *_args, **_kwargs: 0, torch.cuda)  # type: ignore[assignment]

    datamodule = _build_datamodule(config)
    datamodule.num_workers = 0  # Avoid multiprocessing in restricted environments
    datamodule.pin_memory = False
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    frames = batch.get("frames")
    targets = batch.get("targets_norm")
    target_heatmaps = batch.get("target_heatmaps")
    visibility = batch.get("visibility")

    if frames is not None:
        print(f"Loaded batch: frames {tuple(frames.shape)}")
    if targets is not None:
        print(f"targets {tuple(targets.shape)}")
    if target_heatmaps is not None:
        print(f"heatmaps {tuple(target_heatmaps.shape)}")
    if visibility is not None:
        print(f"visibility {tuple(visibility.shape)}")
    if (
        "frame_paths" in batch
        and frames is not None
        and getattr(frames, "dim", None) is not None
        and frames.dim() == 5
        and frames.shape[2] == 3
    ):
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

def _build_datamodule(config: DictConfig):
    data_cfg = config.get("data", {})
    data_name = str(data_cfg.get("name", "ball_detection")).lower()
    if data_name == "patch_embeddings":
        return PatchEmbeddingsDataModule(config)
    return BallDetectionDataModule(config)


@hydra.main(config_path="../../configs", config_name="train_ball_detection", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entry point."""
    pl.seed_everything(config.run.seed)

    _setup_logging(config)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    model_name = str(config.model.name)
    data_name = str(config.data.get("name", "ball_detection")).lower()
    if data_name == "patch_embeddings":
        logging.getLogger(__name__).warning(
            "Using patch_embeddings data: ensure model/handlers accept frames shaped [B, T, N, C]."
        )
    output_dir = Path(config.run.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    if config.run.dry_run:
        run_dry_run(config, output_dir)
        return

    resume_ckpt = resolve_resume_ckpt_path(
        args_resume=None,
        config=config,
        output_dir=output_dir,
    )

    datamodule = _build_datamodule(config)
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
    if isinstance(datamodule.train_sampler, VisibilityCurriculumSampler):
        callbacks.append(CurriculumStepCallback(datamodule.train_sampler))

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if config.run.gpus > 0 else "cpu",
        devices=config.run.gpus if config.run.gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=config.run.fast_dev_run,
        precision=config.training.precision,
    )

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=resume_ckpt)

    if not config.run.fast_dev_run:
        trainer.test(lightning_module, datamodule=datamodule)

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
