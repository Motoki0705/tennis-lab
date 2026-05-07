"""Unified PyTorch Lightning module for PLCS training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.tasks.plcs.models import build_plcs_model
from src.tasks.plcs.utils.pose_geometry import canonical_pose_to_world_pose
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.metrics import PLCSMetrics
from src.utils.schema.player import COCO17_SKELETON

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSLightningModule(BaseLightningModule):
    """Lightning module for unified PLCS I/O training."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        self.model: nn.Module = build_plcs_model(self.config)
        self.predict_canonical_pose = bool(
            ((self.config.get("model", {}) or {}).get("predict_canonical_pose", False))
        )

        loss_cfg_dict = self.config.get("loss", {})
        if loss_cfg_dict:
            loss_cfg = PLCSLossConfig.from_dict(dict(loss_cfg_dict))
        else:
            train_cfg = self.config.get("training", {})
            loss_cfg = PLCSLossConfig(
                position_weight=float(train_cfg.get("position_loss_weight", 1.0)),
                rotation_weight=float(train_cfg.get("rotation_loss_weight", 1.0)),
                canonical_pose_weight=float(train_cfg.get("canonical_pose_weight", 0.0)),
            )
        self.loss_fn = PLCSLoss(config=loss_cfg)

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

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        outputs = self._forward_from_batch(batch)
        human_mask = batch.get("human_mask")

        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            pred_canonical_pose=outputs.get("canonical_pose"),
            target_human_kp_3d=batch.get("human_kp_3d"),
            human_mask=human_mask,
        )

        metrics = self._select_metrics(stage).update(
            outputs["position"],
            outputs["rotation"],
            batch["position"],
            batch["rotation"],
            human_mask=human_mask,
        )

        return losses["total"], {
            **metrics,
            **{f"loss_{k}": float(v.item()) for k, v in losses.items()},
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_error_m", metrics.get("position_error_m", 0.0), prog_bar=True)
        self.log("train/ang_error_deg", metrics.get("angular_error_deg", 0.0), prog_bar=True)
        if "loss_canonical_pose" in metrics:
            self.log("train/loss_canonical_pose", metrics["loss_canonical_pose"])
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos_error_m", metrics.get("position_error_m", 0.0), prog_bar=True)
        self.log("val/ang_error_deg", metrics.get("angular_error_deg", 0.0), prog_bar=True)
        if "loss_canonical_pose" in metrics:
            self.log("val/loss_canonical_pose", metrics["loss_canonical_pose"])

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/pos_error_m", metrics.get("position_error_m", 0.0))
        self.log("test/ang_error_deg", metrics.get("angular_error_deg", 0.0))
        if "loss_canonical_pose" in metrics:
            self.log("test/loss_canonical_pose", metrics["loss_canonical_pose"])

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()

    # ------------------------------------------------------------------
    # Qualitative validation logging
    # ------------------------------------------------------------------

    def _draw_pose_topdown(
        self,
        canvas: np.ndarray,
        pose_xyz: np.ndarray,
        *,
        to_px: Any,
        color: tuple[int, int, int],
    ) -> None:
        points = [to_px(joint[:2]) for joint in pose_xyz]
        for start_idx, end_idx in COCO17_SKELETON:
            cv2.line(canvas, points[start_idx], points[end_idx], color, 2)
        for point in points:
            cv2.circle(canvas, point, 3, color, -1)

    def _render_pose_topdown_canvas(
        self,
        gt_pose_world: np.ndarray,
        pred_pose_world: np.ndarray,
    ) -> np.ndarray:
        if gt_pose_world.ndim == 2:
            gt_pose_world = gt_pose_world[np.newaxis]
            pred_pose_world = pred_pose_world[np.newaxis]

        fig_w, fig_h = 500, 500
        canvas = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 255

        all_xy = np.concatenate(
            [
                gt_pose_world[..., :2].reshape(-1, 2),
                pred_pose_world[..., :2].reshape(-1, 2),
            ],
            axis=0,
        )
        mn = all_xy.min(axis=0)
        mx = all_xy.max(axis=0)
        rng = (mx - mn).clip(1e-3)
        margin = 40

        def to_px(p: np.ndarray) -> tuple[int, int]:
            x = int((p[0] - mn[0]) / rng[0] * (fig_w - 2 * margin) + margin)
            y = int((p[1] - mn[1]) / rng[1] * (fig_h - 2 * margin) + margin)
            return (np.clip(x, 0, fig_w - 1), np.clip(y, 0, fig_h - 1))

        for frame_idx in range(gt_pose_world.shape[0]):
            self._draw_pose_topdown(
                canvas,
                gt_pose_world[frame_idx],
                to_px=to_px,
                color=(0, 180, 0),
            )
            self._draw_pose_topdown(
                canvas,
                pred_pose_world[frame_idx],
                to_px=to_px,
                color=(0, 0, 255),
            )

        cv2.putText(
            canvas,
            "Top-down pose: Green=GT, Red=Pred",
            (5, fig_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        return canvas

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
                pred_world_pose = None
                if self.predict_canonical_pose and "canonical_pose" in out:
                    pred_world_pose = canonical_pose_to_world_pose(
                        out["canonical_pose"],
                        out["position"],
                        out["rotation"],
                    ).cpu().numpy()

            gt_pos = batch["position"].numpy()
            gt_rot = batch["rotation"].numpy()
            gt_world_pose = batch.get("human_kp_3d")

            # Render first sample
            b = 0
            if pred_world_pose is not None and isinstance(gt_world_pose, Tensor):
                canvas = self._render_pose_topdown_canvas(
                    gt_world_pose.numpy()[b],
                    pred_world_pose[b],
                )
                path = artifact_dir / f"plcs_batch{batch_idx:02d}.png"
                cv2.imwrite(str(path), canvas)

                save_image_to_tensorboard(
                    tb_writer,
                    f"qualitative/plcs/batch{batch_idx:02d}",
                    canvas,
                    global_step,
                )
                continue

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
