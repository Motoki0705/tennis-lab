"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_animation
from src.tasks.blcs.configuration import parse_qualitative_rendering
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

    def __init__(self, config: DictConfig) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        self.model = build_blcs_model(self.config)
        self.qualitative_rendering = parse_qualitative_rendering(self.config)

        train_cfg = self.config.training
        self.max_epochs = int(train_cfg.trainer.max_epochs)
        # The gravity prior needs an absolute physical scale: derive the
        # output-frame dt and g from the run config (rally / physics) rather than
        # hard-coding, so a change to output_fps or gravity flows through.
        rally_cfg = self.config.rally
        physics_cfg = self.config.physics
        output_fps = float(rally_cfg.output_fps)
        self.loss_fn = BLCSLoss(
            position_weight=train_cfg.position_loss_weight,
            reprojection_weight=train_cfg.reprojection_loss_weight,
            position_axis_weights=train_cfg.position_axis_weights,
            smoothness_weight=train_cfg.smoothness_loss_weight,
            gravity_weight=train_cfg.gravity_loss_weight,
            smoothness_order=int(train_cfg.smoothness_order),
            smoothness_beta=float(train_cfg.smoothness_beta),
            smoothness_axis_weights=train_cfg.smoothness_axis_weights,
            gravity_beta=float(train_cfg.gravity_beta),
            gravity=float(physics_cfg.gravity),
            frame_dt=1.0 / output_fps,
        )
        gan_enabled = bool(train_cfg.gan.enabled)
        self._initialize_manual_gan(
            discriminator=build_blcs_discriminator(self.config)
            if gan_enabled
            else None,
        )

        metrics_cfg = self.config.metrics
        self.train_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.position_threshold_m,
            endpoint_threshold_m=metrics_cfg.endpoint_threshold_m,
        )
        self.val_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.position_threshold_m,
            endpoint_threshold_m=metrics_cfg.endpoint_threshold_m,
        )
        self.test_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.position_threshold_m,
            endpoint_threshold_m=metrics_cfg.endpoint_threshold_m,
        )

    def _forward_from_batch(
        self, batch: BLCSBatch | BLCSMultiViewBatch
    ) -> dict[str, Tensor]:
        """Forward model from a batch."""
        result: dict[str, Tensor] = self.model(
            ball_uv=batch["ball_uv"],
            court_kp=batch["court_kp"],
            ball_vis=batch.get("ball_vis"),
            ball_mask=batch.get("ball_mask"),
            court_vis=batch.get("court_vis"),
        )
        return result

    def _normalize_loss_mask(
        self, batch: BLCSBatch | BLCSMultiViewBatch
    ) -> Tensor | None:
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

    def test_prediction_payload(
        self, batch: BLCSBatch | BLCSMultiViewBatch, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Ball 3D position predictions + targets to persist for the test split."""
        outputs = result["outputs"]
        payload: dict[str, Any] = {
            "pred_position": outputs["position"],
            "target_position": batch["position_3d"],
        }
        mask = result.get("mask")
        if mask is not None:
            payload["mask"] = mask
        return payload

    def _log_stage_metrics(
        self, stage: str, loss: Tensor, metrics: dict[str, Any]
    ) -> None:
        prog_bar = stage != "test"
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar)
        self.log(
            f"{stage}/pos_error_m",
            metrics["position_error_m"],
            prog_bar=prog_bar,
        )
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
        """Render GT vs predicted 3D ball trajectories using BLCSSceneRenderer."""
        # Deferred imports to avoid circular dependency:
        #   lightning_module <- inference.predictor <- visualization <- lightning_module
        from src.tasks.blcs.visualization.adapters.render_inputs import (
            batch_to_trajectory_arrays,  # noqa: PLC0415
        )
        from src.tasks.blcs.visualization.rendering.scene_renderer import (
            BLCSSceneRenderer,  # noqa: PLC0415
        )

        device = next(self.parameters()).device
        renderer = BLCSSceneRenderer(style=self.qualitative_rendering.style)

        for i, batch in enumerate(batches):
            batch_dev = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                out = self._forward_from_batch(
                    cast("BLCSBatch | BLCSMultiViewBatch", batch_dev)
                )

            gt, pred = batch_to_trajectory_arrays(batch, out, sample_idx=0)

            anim = renderer.create_comparison_animation(gt, pred, view="3d")
            if anim is None:
                continue

            save_qualitative_animation(
                animation=anim,
                artifact_dir=artifact_dir,
                name=f"blcs_batch{i:02d}",
                tb_writer=tb_writer,
                tag=f"qualitative/blcs/batch{i:02d}",
                global_step=global_step,
                fps=self.qualitative_rendering.fps,
            )
