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
from src.tasks.ball_detection.models import (
    build_ball_detection_discriminator,
    build_ball_detection_model,
)
from src.tasks.ball_detection.training.losses import BallDetectionFocalLoss
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.utils.data.heatmaps import heatmaps_to_soft_argmax

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallDetectionLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for training ball detection.

    Supervised training optimizes heatmap losses. When ``training.gan.enabled``
    is set, ball coordinate sequences extracted from predicted heatmaps via
    differentiable soft-argmax are scored by a trajectory discriminator against
    ground-truth coordinate sequences, and the binary real/fake signal is fed
    back to the heatmap model as an adversarial loss.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        loss_cfg = self.config.get("loss", {})
        metrics_cfg = self.config.get("metrics", {})

        self.model = build_ball_detection_model(self.config)

        self.loss_fn = BallDetectionFocalLoss(loss_cfg)

        train_cfg = self.config.get("training", {})
        gan_cfg = train_cfg.get("gan", {}) or {}
        gan_enabled = bool(gan_cfg.get("enabled", False))
        self.gan_soft_argmax_temperature = float(
            gan_cfg.get("soft_argmax_temperature", 1.0)
        )
        self._initialize_manual_gan(
            discriminator=(
                build_ball_detection_discriminator(self.config) if gan_enabled else None
            ),
        )

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

    def _predict_heatmap_logits(self, images: Tensor, target_size_hw: tuple[int, int]) -> Tensor:
        """Predict per-frame heatmap logits resized to the target heatmap size."""
        model_cfg = self.config.get("model", {})
        model_input = to_model_input(images, model_cfg)

        logits = self.model(model_input)

        # Squeeze channel dim: (B, 1, T, Hh, Wh) -> (B, T, Hh, Wh)
        logits = logits.squeeze(1)

        # Interpolate if model output size != target heatmap size
        if logits.shape[-2:] != target_size_hw:
            b, t = logits.shape[:2]
            logits_flat = logits.reshape(b * t, 1, *logits.shape[-2:])
            logits_flat = F.interpolate(
                logits_flat,
                size=target_size_hw,
                mode="bilinear",
                align_corners=False,
            )
            logits = logits_flat.reshape(b, t, *target_size_hw)
        return logits

    def _extract_gt_trajectory(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Extract the primary ball trajectory and its visibility mask.

        Returns:
            Tuple of:
                - ball_xy: Normalized ``(B, T, 2)`` coordinates in ``(x, y)``.
                - mask: Boolean ``(B, T)`` mask of frames with a visible ball.
        """
        coords = batch["coords"]  # (B, T, K, 2) in original image pixels
        visibility = batch["visibility"]  # (B, T, K)
        original_size = batch["original_size"]  # (B, 2) as (width, height)

        # Pick the first visible instance per frame (argmax of a 0/1 mask
        # returns the first maximal entry).
        first_visible = visibility.argmax(dim=-1)  # (B, T)
        gather_index = first_visible[..., None, None].expand(-1, -1, 1, 2)
        ball_xy = coords.gather(2, gather_index).squeeze(2)  # (B, T, 2)

        scale = (original_size - 1.0).clamp(min=1.0)  # (B, 2)
        ball_xy = ball_xy / scale[:, None, :]
        mask = visibility.amax(dim=-1) > 0.5
        return ball_xy, mask

    def _compute_supervised_result(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Any]:
        """Compute forward pass, supervised loss, metrics, and GAN sequences."""
        target_heatmaps = batch["heatmaps"]
        logits = self._predict_heatmap_logits(
            batch["images"],
            target_heatmaps.shape[-2:],
        )

        loss = self.loss_fn(logits, target_heatmaps)
        pred_heatmaps = torch.sigmoid(logits)

        self._select_metrics(stage).update(
            pred_heatmaps,
            batch["coords"],
            batch["visibility"],
            batch["original_size"],
        )

        gan_real, gan_mask = self._extract_gt_trajectory(batch)
        gan_fake = heatmaps_to_soft_argmax(
            logits,
            temperature=self.gan_soft_argmax_temperature,
        )

        return {
            "loss": loss,
            "metrics": {},
            "pred_heatmaps": pred_heatmaps,
            "gan_fake": gan_fake,
            "gan_real": gan_real,
            "gan_mask": gan_mask,
        }

    def _select_metrics(self, stage: str) -> BallDetectionMetrics:
        """Return the metrics object for the current stage."""
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _metric_tracker_for_stage(self, stage: str) -> BallDetectionMetrics:
        return self._select_metrics(stage)

    def _flush_stage_metrics(self, stage: str) -> None:
        tracker = self._metric_tracker_for_stage(stage)
        metrics = tracker.compute()
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=(stage == "val" and name == "f1"),
            )
        tracker.reset()

    def _log_stage_metrics(self, stage: str, loss: Tensor, metrics: dict[str, Any]) -> None:
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        if stage == "train" and self.gan_enabled:
            self.log("train/gan_weight", float(self.current_gan_weight))
            self.log("train/gan_phase_active", float(self.gan_phase_active))
            if "loss_gan_generator" in metrics:
                self.log("train/loss_gan_generator", metrics["loss_gan_generator"])
            if "loss_gan_discriminator" in metrics:
                self.log("train/loss_gan_discriminator", metrics["loss_gan_discriminator"])

    def configure_optimizers(self) -> Any:
        """Configure generator/discriminator optimizers through the shared GAN helper."""
        return self.configure_gan_optimizers(self.model.parameters())

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
            coords_gt = batch["coords"]  # (B, T, K, 2)
            visibility = batch["visibility"]  # (B, T, K)

            with torch.no_grad():
                logits = self._predict_heatmap_logits(images, heatmaps_gt.shape[-2:])
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

                visible_coords = coords_gt[
                    b_idx,
                    t_idx,
                    visibility[b_idx, t_idx] > 0.5,
                ]
                for coord in visible_coords:
                    cx = int(coord[0].item())
                    cy = int(coord[1].item())
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
