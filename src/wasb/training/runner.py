"""WASB Training Runner.

Extends BaseTrainingRunner with WASB-specific overrides for:
- DataModule selection (BallDetectionDataModule / PatchEmbeddingsDataModule)
- Model and io_handlers construction
- Resume checkpoint resolution
- Curriculum learning callbacks
- Dry-run visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from src.base.training.runner import BaseTrainingRunner
from src.wasb.data.ball_detection_datamodule import BallDetectionDataModule
from src.wasb.data.curriculum_sampling import (
    CurriculumStepCallback,
    VisibilityCurriculumSampler,
)
from src.wasb.data.patch_embeddings_datamodule import PatchEmbeddingsDataModule
from src.wasb.models import build_model
from src.wasb.training import WASBLightningModule
from src.wasb.utils.checkpoint import resolve_resume_ckpt_path

if TYPE_CHECKING:
    from pytorch_lightning.loggers import TensorBoardLogger


class WASBTrainingRunner(BaseTrainingRunner):
    """Training runner for WASB ball detection models."""

    # ---- required overrides ----

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build WASB datamodule based on config."""
        data_cfg = config.get("data", {})
        data_name = str(data_cfg.get("name", "ball_detection")).lower()
        if data_name == "patch_embeddings":
            return PatchEmbeddingsDataModule(config)
        return BallDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build WASBLightningModule with model and io_handlers."""
        model, io_handlers = build_model(config)

        # Load backbone checkpoint if specified
        model_cfg = config.get("model")
        backbone_ckpt = None
        if model_cfg and hasattr(model_cfg, "get"):
            backbone_ckpt = model_cfg.get("backbone_checkpoint")
        if backbone_ckpt and hasattr(model, "load_backbone_checkpoint"):
            model.load_backbone_checkpoint(backbone_ckpt)

        return WASBLightningModule(
            config,
            model=model,
            steps_per_epoch=steps_per_epoch,
            io_handlers=io_handlers,
        )

    # ---- customization hooks ----

    def prepare_output_dir(self, config: Any) -> Path:
        """Include model name in output directory."""
        base_dir = Path(self._ensure_absolute(str(config.run.output_dir)))
        model_name = str(config.model.name)
        return base_dir / model_name

    def resolve_resume(self, config: Any, output_dir: Path) -> str | None:
        """Use WASB-specific resume checkpoint resolution."""
        return resolve_resume_ckpt_path(
            args_resume=None,
            config=config,
            output_dir=output_dir,
        )

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None = None,
    ) -> int | None:
        """Compute steps per epoch from train dataloader."""
        if train_loader is not None:
            return len(train_loader)

        # Setup datamodule to get train loader length
        # Note: datamodule.setup should be idempotent if called multiple times.
        if hasattr(datamodule, "setup"):
            datamodule.setup(stage="fit")
        loader = datamodule.train_dataloader()
        return len(loader)

    def checkpoint_prefix(self, config: Any) -> str:
        """Use 'wasb' as checkpoint filename prefix."""
        return "wasb"

    def early_stopping_enabled(self, config: Any) -> bool:
        """Disable early stopping by default for WASB."""
        return False

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        """Add precision setting from config."""
        kwargs: dict[str, Any] = {}
        precision = getattr(config.training, "precision", None)
        if precision is not None:
            kwargs["precision"] = precision
        # Disable deterministic mode for WASB (some ops may not support it)
        kwargs["deterministic"] = False
        return kwargs

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        """Add CurriculumStepCallback if using VisibilityCurriculumSampler."""
        callbacks: list[Any] = []
        if hasattr(datamodule, "train_sampler") and isinstance(
            datamodule.train_sampler, VisibilityCurriculumSampler
        ):
            callbacks.append(CurriculumStepCallback(datamodule.train_sampler))
        return callbacks

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Save sample frame/heatmap visuals for inspection."""
        if not isinstance(batch, dict):
            return

        frames = batch.get("frames")
        frame_paths = batch.get("frame_paths")
        target_heatmaps = batch.get("target_heatmaps")

        # Only process if we have frames in expected format [B, T, C, H, W] with C=3
        if (
            frames is None
            or target_heatmaps is None
            or frame_paths is None
            or not hasattr(frames, "dim")
            or frames.dim() != 5
            or frames.shape[2] != 3
        ):
            return

        dry_run_dir = output_dir / "dry_run"
        self._save_sample_visuals(batch, dry_run_dir)

    def _save_sample_visuals(self, batch: dict, out_dir: Path) -> None:
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

        # Recover the per-sample frame paths for the selected frames.
        selected_paths: list[str] = []
        for idx in range(len(heatmaps)):
            frame_idx = frame_start + idx
            if frame_idx < len(frame_paths):
                # frame_paths is list-of-lists where outer index is frame position, inner is batch.
                selected_paths.append(frame_paths[frame_idx][0])
        if selected_paths:
            (out_dir / "frame_paths.txt").write_text("\n".join(selected_paths))
