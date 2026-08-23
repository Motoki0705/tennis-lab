"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.model_io import (
    validate_checkpoint_court_coordinate_contract,
    write_checkpoint_court_coordinate_contract,
)
from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_animation
from src.tasks.blcs.configuration import (
    parse_court_coordinate_normalization,
    parse_qualitative_rendering,
    resolve_position_huber_beta,
)
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    TrajectoryBoundModelIO,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.models import build_blcs_discriminator
from src.tasks.blcs.training.losses import BLCSLoss
from src.tasks.blcs.training.metrics import BLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for BLCS models.

    This module supports both single-view and multiview BLCS training.
    """

    def __init__(
        self,
        config: DictConfig,
        *,
        model_io: TrajectoryBoundModelIO,
    ) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        self.model_io = model_io
        self.model = model_io.model
        self.io_adapter = cast("TrajectoryModelIOAdapter", model_io.adapter)
        self.court_coordinate_normalization = (
            parse_court_coordinate_normalization(self.config)
        )
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
            position_beta=resolve_position_huber_beta(
                self.court_coordinate_normalization,
                legacy_v1_beta=float(train_cfg.position_huber_beta_v1),
                v2_transition_m=float(
                    train_cfg.position_huber_transition_m_v2
                ),
            ),
            normalization=self.court_coordinate_normalization,
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
            normalization=self.court_coordinate_normalization,
        )
        self.val_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.position_threshold_m,
            endpoint_threshold_m=metrics_cfg.endpoint_threshold_m,
            normalization=self.court_coordinate_normalization,
        )
        self.test_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.position_threshold_m,
            endpoint_threshold_m=metrics_cfg.endpoint_threshold_m,
            normalization=self.court_coordinate_normalization,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the exact normalization contract beside model state."""
        write_checkpoint_court_coordinate_contract(
            checkpoint,
            self.court_coordinate_normalization,
            location="BLCS checkpoint",
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject checkpoint/runtime normalization mismatches before weights."""
        validate_checkpoint_court_coordinate_contract(
            checkpoint,
            self.court_coordinate_normalization,
            location="BLCS checkpoint",
        )

    def _forward_from_batch(
        self, batch: Mapping[str, object]
    ) -> BLCSTrajectoryPrediction:
        """Delegate a validated model invocation to the bound adapter."""
        prediction: BLCSTrajectoryPrediction = self.model_io.run(batch)
        return prediction

    def _compute_supervised_result(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> dict[str, Any]:
        """Compute forward pass, supervised losses, and metrics."""
        prepared = self.io_adapter.build_training_batch(batch)
        outputs = self.model_io.decode_output(self.model_io.execute_call(prepared.call))

        losses = self.loss_fn(
            pred_position=outputs.position,
            target_position=prepared.position,
            mask=prepared.loss_mask,
            target_uv=prepared.target_uv,
            target_vis=prepared.target_vis,
            camera_R=prepared.camera_R,
            camera_C=prepared.camera_C,
            camera_f=prepared.camera_f,
            camera_cx=prepared.camera_cx,
            camera_cy=prepared.camera_cy,
            camera_w=prepared.camera_w,
            camera_h=prepared.camera_h,
        )

        metrics = self._metric_tracker_for_stage(stage).update(
            outputs.position,
            prepared.position,
            prepared.loss_mask,
        )

        return {
            "loss": losses["total"],
            "losses": losses,
            "metrics": {
                **metrics,
                **{f"loss_{k}": v.item() for k, v in losses.items()},
            },
            "outputs": outputs,
            "mask": prepared.loss_mask,
            "gan_fake": outputs.position,
            "gan_real": prepared.position,
            "gan_padding_mask": ~prepared.loss_mask,
        }

    def test_prediction_payload(
        self, batch: Mapping[str, object], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Ball 3D position predictions + targets to persist for the test split."""
        outputs = cast("BLCSTrajectoryPrediction", result["outputs"])
        target = self.io_adapter.build_training_batch(batch)
        payload: dict[str, Any] = {
            "pred_position": outputs.position,
            "target_position": target.position,
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
                out = self._forward_from_batch(batch_dev)

            gt, pred = self.io_adapter.trajectory_arrays(
                batch,
                out,
                sample_index=0,
            )
            gt_m = self.court_coordinate_normalization.denormalize_position(gt)
            pred_m = self.court_coordinate_normalization.denormalize_position(pred)
            if not isinstance(gt_m, np.ndarray) or not isinstance(
                pred_m, np.ndarray
            ):
                raise TypeError(
                    "BLCS qualitative denormalization returned a non-array."
                )

            anim = renderer.create_comparison_animation(
                gt_m,
                pred_m,
                view="3d",
                events=[],
            )
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
