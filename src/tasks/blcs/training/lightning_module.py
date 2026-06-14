"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import Tensor

from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.tasks.blcs.data.types import BLCSBatch, BLCSMultiViewBatch
from src.tasks.blcs.models import build_blcs_discriminator, build_blcs_model
from src.tasks.blcs.training.losses import BLCSLoss
from src.tasks.blcs.training.metrics import BLCSMetrics
from src.utils.tensor_utils import normalize_padding_mask

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for BLCS models.

    This module supports both single-view and multiview BLCS training.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        self.model = build_blcs_model(self.config)

        train_cfg = self.config.get("training", {})
        trainer_cfg = train_cfg.get("trainer", {}) or {}
        self.max_epochs = int(trainer_cfg.get("max_epochs", self.max_epochs))
        self.loss_fn = BLCSLoss(
            position_weight=train_cfg.get("position_loss_weight", 1.0),
            velocity_weight=train_cfg.get("velocity_loss_weight", 0.1),
            smoothness_weight=train_cfg.get("smoothness_loss_weight", 0.05),
            reprojection_weight=train_cfg.get("reprojection_loss_weight", 0.0),
            uv_velocity_weight=train_cfg.get("uv_velocity_loss_weight", 0.0),
        )
        gan_enabled = bool((train_cfg.get("gan", {}) or {}).get("enabled", False))
        self._initialize_manual_gan(
            discriminator=build_blcs_discriminator(self.config) if gan_enabled else None,
        )

        metrics_cfg = self.config.get("metrics", {})
        self.train_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.get("position_threshold_m", 0.3),
            endpoint_threshold_m=metrics_cfg.get("endpoint_threshold_m", 0.5),
        )
        self.val_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.get("position_threshold_m", 0.3),
            endpoint_threshold_m=metrics_cfg.get("endpoint_threshold_m", 0.5),
        )
        self.test_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.get("position_threshold_m", 0.3),
            endpoint_threshold_m=metrics_cfg.get("endpoint_threshold_m", 0.5),
        )

    def _forward_from_batch(self, batch: BLCSBatch | BLCSMultiViewBatch) -> dict[str, Tensor]:
        """Forward model from a batch."""
        return self.model(
            ball_uv=batch["ball_uv"],
            court_kp=batch["court_kp"],
            ball_vis=batch.get("ball_vis"),
            ball_mask=batch.get("ball_mask"),
            court_vis=batch.get("court_vis"),
        )

    def _normalize_loss_mask(self, batch: BLCSBatch | BLCSMultiViewBatch) -> Tensor | None:
        """Normalize loss/metric mask to shape (B, T)."""
        return normalize_padding_mask(batch.get("ball_mask"))

    def _compute_supervised_result(
        self,
        batch: BLCSBatch | BLCSMultiViewBatch,
        stage: str,
    ) -> dict[str, Any]:
        """Compute forward pass, supervised losses, and metrics."""
        outputs = self._forward_from_batch(batch)
        mask = self._normalize_loss_mask(batch)

        losses = self.loss_fn(
            pred_position=outputs["position"],
            target_position=batch.get("position_3d"),
            mask=mask,
            target_uv=batch.get("ball_uv_target", batch.get("ball_uv")),
            target_vis=batch.get("ball_vis_target", batch.get("ball_vis")),
            camera_R=batch.get("camera_R"),
            camera_C=batch.get("camera_C"),
            camera_f=batch.get("camera_f"),
            camera_cx=batch.get("camera_cx"),
            camera_cy=batch.get("camera_cy"),
            camera_w=batch.get("camera_w"),
            camera_h=batch.get("camera_h"),
        )

        metrics = self._metric_tracker_for_stage(stage).update(
            outputs["position"],
            batch["position_3d"],
            mask,
        )

        return {
            "loss": losses["total"],
            "losses": losses,
            "metrics": {
                **metrics,
                **{f"loss_{k}": v.item() for k, v in losses.items()},
            },
            "outputs": outputs,
            "mask": mask,
            "gan_fake": outputs["position"],
            "gan_real": batch["position_3d"],
            "gan_mask": mask,
        }

    def _log_stage_metrics(self, stage: str, loss: Tensor, metrics: dict[str, Any]) -> None:
        prog_bar = stage != "test"
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar)
        self.log(f"{stage}/pos_error_m", metrics.get("position_error_m", 0), prog_bar=prog_bar)
        self._log_gan_metrics(stage, metrics)

    def _metric_tracker_for_stage(self, stage: str) -> BLCSMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

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
        """Render GT vs predicted 3D ball trajectories in top-down (X-Y) and side (X-Z) views."""
        device = next(self.parameters()).device

        for batch_idx, batch in enumerate(batches):
            batch_dev = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                out = self._forward_from_batch(batch_dev)
                pred_pos = out["position"].cpu().numpy()  # (B, T, 3)

            gt_pos = batch["position_3d"].numpy()  # (B, T, 3)
            mask = self._normalize_loss_mask(batch)
            if mask is not None:
                mask_np = mask.numpy()
            else:
                mask_np = np.ones(gt_pos.shape[:2], dtype=np.float32)

            # Render first sample
            b = 0
            gt = gt_pos[b]  # (T, 3)
            pred = pred_pos[b]  # (T, 3)
            m = mask_np[b] > 0  # (T,)

            fig_w, fig_h = 400, 400
            # Top-down: X vs Y
            canvas_top = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255
            # Side: X vs Z
            canvas_side = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

            def _normalize_and_draw(
                canvas: np.ndarray, gt_2d: np.ndarray, pred_2d: np.ndarray, valid: np.ndarray
            ) -> None:
                all_pts = np.concatenate([gt_2d[valid], pred_2d[valid]], axis=0)
                if len(all_pts) == 0:
                    return
                mn = all_pts.min(axis=0)
                mx = all_pts.max(axis=0)
                rng = (mx - mn).clip(1e-3)
                margin = 30
                canvas_h, canvas_w = canvas.shape[:2]

                def to_px(p: np.ndarray) -> tuple[int, int]:
                    x = int((p[0] - mn[0]) / rng[0] * (canvas_w - 2 * margin) + margin)
                    y = int((p[1] - mn[1]) / rng[1] * (canvas_h - 2 * margin) + margin)
                    return (np.clip(x, 0, canvas_w - 1), np.clip(y, 0, canvas_h - 1))

                for t in range(len(gt_2d)):
                    if not valid[t]:
                        continue
                    cv2.circle(canvas, to_px(gt_2d[t]), 3, (0, 180, 0), -1)
                    cv2.circle(canvas, to_px(pred_2d[t]), 3, (0, 0, 255), -1)

            _normalize_and_draw(canvas_top, gt[:, :2], pred[:, :2], m)
            cv2.putText(canvas_top, "Top-down (X-Y) Green=GT Red=Pred", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            _normalize_and_draw(canvas_side, gt[:, [0, 2]], pred[:, [0, 2]], m)
            cv2.putText(canvas_side, "Side (X-Z) Green=GT Red=Pred", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            panel = np.concatenate([canvas_top, canvas_side], axis=1)

            path = artifact_dir / f"blcs_batch{batch_idx:02d}.png"
            cv2.imwrite(str(path), panel)

            save_image_to_tensorboard(
                tb_writer,
                f"qualitative/blcs/batch{batch_idx:02d}",
                panel,
                global_step,
            )
