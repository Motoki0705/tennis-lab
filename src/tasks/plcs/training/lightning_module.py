"""Unified PyTorch Lightning module for PLCS training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.tasks.plcs.models import build_plcs_discriminator, build_plcs_model
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.metrics import PLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for unified PLCS I/O training."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        self.model: nn.Module = build_plcs_model(self.config)

        loss_cfg_dict = self.config.get("loss", {})
        if loss_cfg_dict:
            loss_cfg = PLCSLossConfig.from_dict(dict(loss_cfg_dict))
        else:
            train_cfg = self.config.get("training", {})
            loss_cfg = PLCSLossConfig(
                position_weight=float(train_cfg.get("position_loss_weight", 1.0)),
                rotation_weight=float(train_cfg.get("rotation_loss_weight", 1.0)),
            )
        self.loss_fn = PLCSLoss(config=loss_cfg)
        gan_enabled = bool(((self.config.get("training", {}) or {}).get("gan", {}) or {}).get("enabled", False))
        self._initialize_manual_gan(
            discriminator=build_plcs_discriminator(self.config) if gan_enabled else None,
        )

        metrics_cfg = self.config.get("metrics", {})
        self.train_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
            angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
        )
        self.val_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
            angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
        )
        self.test_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
            angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
        )

    def _normalize_frame_mask(self, human_mask: Tensor | None) -> Tensor | None:
        if human_mask is None:
            return None
        if human_mask.dim() == 1:
            return human_mask > 0
        if human_mask.dim() == 2:
            return human_mask > 0
        if human_mask.dim() == 3:
            return (human_mask > 0).any(dim=1)
        if human_mask.dim() == 4:
            return (human_mask > 0).any(dim=1).any(dim=-1)
        raise ValueError(
            "human_mask must be (B,), (B,T), (B,N,T), or (B,N,T,J), "
            f"got shape {tuple(human_mask.shape)}"
        )

    def _pose_sequence(self, position: Tensor, rotation: Tensor) -> Tensor:
        if position.ndim == 2:
            position = position.unsqueeze(1)
            rotation = rotation.unsqueeze(1)
        return torch.cat([position, rotation], dim=-1)

    def _forward_from_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(
            human_kp=batch["human_kp"],
            court_kp=batch["court_kp"],
            human_vis=batch.get("human_vis"),
            human_mask=batch.get("human_mask"),
            court_vis=batch.get("court_vis"),
        )

    def _select_metrics(self, stage: str) -> PLCSMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _compute_supervised_result(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Any]:
        outputs = self._forward_from_batch(batch)
        human_mask = batch.get("human_mask")
        frame_mask = self._normalize_frame_mask(human_mask)

        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            human_mask=human_mask,
        )

        metrics = self._select_metrics(stage).update(
            outputs["position"],
            outputs["rotation"],
            batch["position"],
            batch["rotation"],
            human_mask=human_mask,
        )

        return {
            "loss": losses["total"],
            "metrics": {
                **metrics,
                **{f"loss_{k}": float(v.item()) for k, v in losses.items()},
            },
            "outputs": outputs,
            "gan_fake": self._pose_sequence(outputs["position"], outputs["rotation"]),
            "gan_real": self._pose_sequence(batch["position"], batch["rotation"]),
            "gan_mask": frame_mask,
        }

    def _log_stage_metrics(self, stage: str, loss: Tensor, metrics: dict[str, Any]) -> None:
        prog_bar = stage != "test"
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar)
        self.log(f"{stage}/pos_error_m", metrics.get("position_error_m", 0.0), prog_bar=prog_bar)
        self.log(
            f"{stage}/ang_error_deg",
            metrics.get("angular_error_deg", 0.0),
            prog_bar=prog_bar,
        )
        if stage == "train" and self.gan_enabled:
            self.log("train/gan_weight", float(self.current_gan_weight))
            self.log("train/gan_phase_active", float(self.gan_phase_active))
            if "loss_gan_generator" in metrics:
                self.log("train/loss_gan_generator", metrics["loss_gan_generator"])
            if "loss_gan_discriminator" in metrics:
                self.log("train/loss_gan_discriminator", metrics["loss_gan_discriminator"])

    def _metric_tracker_for_stage(self, stage: str) -> PLCSMetrics:
        return self._select_metrics(stage)

    def configure_optimizers(self) -> Any:
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
        """Render GT vs predicted player positions in top-down view with orientation arrows."""
        device = next(self.parameters()).device

        for batch_idx, batch in enumerate(batches):
            batch_dev = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                out = self._forward_from_batch(batch_dev)
                pred_pos = out["position"].cpu().numpy()  # (B, [T], 3)
                pred_rot = out["rotation"].cpu().numpy()  # (B, [T], 2)

            gt_pos = batch["position"].numpy()
            gt_rot = batch["rotation"].numpy()

            # Render first sample
            b = 0
            gp = gt_pos[b]  # ([T], 3) or (3,)
            pp = pred_pos[b]
            gr = gt_rot[b]  # ([T], 2) or (2,)
            pr = pred_rot[b]

            # Handle single-frame (no T dim) vs sequence
            if gp.ndim == 1:
                gp = gp[np.newaxis]
                pp = pp[np.newaxis]
                gr = gr[np.newaxis]
                pr = pr[np.newaxis]

            T = gp.shape[0]
            fig_w, fig_h = 500, 500
            canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

            # Use X-Y as top-down
            all_xy = np.concatenate([gp[:, :2], pp[:, :2]], axis=0)
            mn = all_xy.min(axis=0)
            mx = all_xy.max(axis=0)
            rng = (mx - mn).clip(1e-3)
            margin = 40

            def to_px(p: np.ndarray) -> tuple[int, int]:
                x = int((p[0] - mn[0]) / rng[0] * (fig_w - 2 * margin) + margin)
                y = int((p[1] - mn[1]) / rng[1] * (fig_h - 2 * margin) + margin)
                return (np.clip(x, 0, fig_w - 1), np.clip(y, 0, fig_h - 1))

            arrow_len = 20

            for t in range(T):
                # GT: green
                pt_gt = to_px(gp[t, :2])
                cv2.circle(canvas, pt_gt, 4, (0, 180, 0), -1)
                dx_gt = int(arrow_len * float(gr[t, 0]))
                dy_gt = int(arrow_len * float(gr[t, 1]))
                cv2.arrowedLine(canvas, pt_gt, (pt_gt[0] + dx_gt, pt_gt[1] + dy_gt), (0, 180, 0), 2)

                # Pred: red
                pt_pred = to_px(pp[t, :2])
                cv2.circle(canvas, pt_pred, 4, (0, 0, 255), -1)
                dx_pr = int(arrow_len * float(pr[t, 0]))
                dy_pr = int(arrow_len * float(pr[t, 1]))
                cv2.arrowedLine(canvas, pt_pred, (pt_pred[0] + dx_pr, pt_pred[1] + dy_pr), (0, 0, 255), 2)

            cv2.putText(canvas, "Top-down: Green=GT, Red=Pred (arrows=orientation)", (5, fig_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            path = artifact_dir / f"plcs_batch{batch_idx:02d}.png"
            cv2.imwrite(str(path), canvas)

            save_image_to_tensorboard(
                tb_writer,
                f"qualitative/plcs/batch{batch_idx:02d}",
                canvas,
                global_step,
            )
