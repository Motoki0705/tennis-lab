"""PyTorch Lightning module for ball detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.ball_detection.data.argumentation import (
    denormalize_tensor_images_imagenet,
)
from src.tasks.ball_detection.data.utils.input_adapter import to_model_input
from src.tasks.ball_detection.models import build_ball_detection_model
from src.tasks.ball_detection.training.losses import BallDetectionFocalLoss
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallDetectionLightningModule(BaseLightningModule):
    """Lightning module for training ball detection.

    Inherits optimizer/scheduler logic from
    :class:`~src.tasks.base.training.lightning_module.BaseLightningModule`.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)
        self.save_hyperparameters()

        loss_cfg = self.config.get("loss", {})
        metrics_cfg = self.config.get("metrics", {})

        self.model = build_ball_detection_model(self.config)

        self.loss_fn = BallDetectionFocalLoss(loss_cfg)

        self.train_metrics = BallDetectionMetrics(
            peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
            ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
            nms_kernel=int(metrics_cfg.get("nms_kernel", 9)),
            max_predictions_per_frame=int(
                metrics_cfg.get("max_predictions_per_frame", 8)
            ),
        )
        self.val_metrics = BallDetectionMetrics(
            peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
            ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
            nms_kernel=int(metrics_cfg.get("nms_kernel", 9)),
            max_predictions_per_frame=int(
                metrics_cfg.get("max_predictions_per_frame", 8)
            ),
        )
        self.test_metrics = BallDetectionMetrics(
            peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
            ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
            nms_kernel=int(metrics_cfg.get("nms_kernel", 9)),
            max_predictions_per_frame=int(
                metrics_cfg.get("max_predictions_per_frame", 8)
            ),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            images: Input tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Logits of shape ``(B, 1, T, Hh, Wh)``.
        """
        return self.model(images)

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        """Shared computation for train/val/test steps."""
        images = batch["images"]
        target_heatmaps = batch["heatmaps"]

        model_cfg = self.config.get("model", {})
        model_input = to_model_input(images, model_cfg)

        logits = self.model(model_input)

        # Squeeze channel dim: (B, 1, T, Hh, Wh) -> (B, T, Hh, Wh)
        logits = logits.squeeze(1)

        # Interpolate if model output size != target heatmap size
        if logits.shape[-2:] != target_heatmaps.shape[-2:]:
            b, t = logits.shape[:2]
            logits_flat = logits.reshape(b * t, 1, *logits.shape[-2:])
            logits_flat = F.interpolate(
                logits_flat,
                size=target_heatmaps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            logits = logits_flat.reshape(b, t, *target_heatmaps.shape[-2:])

        loss = self.loss_fn(logits, target_heatmaps)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        pred_heatmaps = torch.sigmoid(logits)

        return {
            "loss": loss,
            "pred_heatmaps": pred_heatmaps,
            "target_coords": batch["coords"],
            "target_visibility": batch["visibility"],
            "target_instance_coords": batch["instance_coords"],
            "target_instance_visibility": batch["instance_visibility"],
            "original_size": batch["original_size"],
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        outputs = self._shared_step(batch, "train")
        self.train_metrics.update(
            outputs["pred_heatmaps"],
            outputs["target_coords"],
            outputs["target_visibility"],
            outputs["original_size"],
            outputs["target_instance_coords"],
            outputs["target_instance_visibility"],
        )
        return outputs["loss"]

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self._shared_step(batch, "val")
        self.val_metrics.update(
            outputs["pred_heatmaps"],
            outputs["target_coords"],
            outputs["target_visibility"],
            outputs["original_size"],
            outputs["target_instance_coords"],
            outputs["target_instance_visibility"],
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=(name == "f1"))
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Test step."""
        outputs = self._shared_step(batch, "test")
        self.test_metrics.update(
            outputs["pred_heatmaps"],
            outputs["target_coords"],
            outputs["target_visibility"],
            outputs["original_size"],
            outputs["target_instance_coords"],
            outputs["target_instance_visibility"],
        )

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()

    # ------------------------------------------------------------------
    # Qualitative validation logging
    # ------------------------------------------------------------------

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        """Render ball detection heatmap overlays for qualitative inspection."""
        device = next(self.parameters()).device

        for batch_idx, batch in enumerate(batches):
            images = batch["images"].to(device)
            heatmaps_gt = batch["heatmaps"]  # (B, T, Hh, Wh)
            coords_gt = batch["coords"]  # (B, T, 2)
            visibility = batch["visibility"]  # (B, T)

            model_cfg = self.config.get("model", {})
            model_input = to_model_input(images, model_cfg)

            with torch.no_grad():
                logits = self.model(model_input).squeeze(1)  # (B, T, Hh, Wh)
                if logits.shape[-2:] != heatmaps_gt.shape[-2:]:
                    b, t = logits.shape[:2]
                    logits_flat = logits.reshape(b * t, 1, *logits.shape[-2:])
                    logits_flat = F.interpolate(
                        logits_flat,
                        size=heatmaps_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    logits = logits_flat.reshape(b, t, *heatmaps_gt.shape[-2:])
                pred_heatmaps = torch.sigmoid(logits).cpu()

            # Render the first sample as a 3-row temporal contact sheet.
            b_idx = 0
            normalize_cfg = dict(
                self.config.get("data", {})
                .get("augmentation", {})
                .get("normalize_imagenet", {})
                or {}
            )
            frames = images[b_idx].detach().cpu()  # (T, C, H, W)
            if bool(normalize_cfg.get("enabled", False)):
                frames = denormalize_tensor_images_imagenet(
                    frames,
                    mean=normalize_cfg.get("mean", (0.485, 0.456, 0.406)),
                    std=normalize_cfg.get("std", (0.229, 0.224, 0.225)),
                )
            frames = frames.clamp(0, 1)

            rgb_row: list[np.ndarray] = []
            gt_row: list[np.ndarray] = []
            pred_row: list[np.ndarray] = []
            for t_idx in range(images.shape[1]):
                frame_rgb = frames[t_idx].permute(1, 2, 0).numpy()
                frame_uint8 = (frame_rgb * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                h, w = frame_bgr.shape[:2]

                gt_hm = heatmaps_gt[b_idx, t_idx].numpy()
                pred_hm = pred_heatmaps[b_idx, t_idx].numpy()
                gt_cm = cv2.applyColorMap(
                    cv2.resize((gt_hm * 255).astype(np.uint8), (w, h)),
                    cv2.COLORMAP_JET,
                )
                pred_cm = cv2.applyColorMap(
                    cv2.resize((pred_hm * 255).astype(np.uint8), (w, h)),
                    cv2.COLORMAP_JET,
                )

                if visibility[b_idx, t_idx] > 0:
                    cx = int(coords_gt[b_idx, t_idx, 0].item())
                    cy = int(coords_gt[b_idx, t_idx, 1].item())
                    cv2.circle(frame_bgr, (cx, cy), 5, (0, 255, 0), 2)

                rgb_row.append(frame_bgr)
                gt_row.append(gt_cm)
                pred_row.append(pred_cm)

            panel = np.concatenate(
                [
                    np.concatenate(rgb_row, axis=1),
                    np.concatenate(gt_row, axis=1),
                    np.concatenate(pred_row, axis=1),
                ],
                axis=0,
            )

            # Save artifact
            path = artifact_dir / f"ball_batch{batch_idx:02d}.png"
            cv2.imwrite(str(path), panel)

            # Log to TensorBoard
            save_image_to_tensorboard(
                tb_writer,
                f"qualitative/ball_detection/batch{batch_idx:02d}",
                panel,
                global_step,
            )
