"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from src.tasks.base.model_io import (
    validate_model_artifact_court_keypoint_contract,
    write_model_artifact_court_keypoint_contract,
)
from src.tasks.base.model_io.court_keypoint_contract import (
    validate_neural_court_observation_order,
)
from src.tasks.base.training.gan_training import (
    ManualGANSupportMixin,
    loss_component_metrics,
)
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.metric_logging import (
    MetricLoggingContract,
    StageMetricContract,
)
from src.tasks.base.training.qualitative_saving import save_qualitative_animation
from src.tasks.blcs.configuration import (
    parse_court_keypoint_contract,
    parse_qualitative_rendering,
)
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    TrajectoryBoundModelIO,
    TrajectoryModelIOAdapter,
    blcs_reference_metadata_from_batch,
)
from src.tasks.blcs.models import build_blcs_discriminator
from src.tasks.blcs.training.losses import PHYSICAL_TERMS, BLCSLoss
from src.tasks.blcs.training.metrics import BLCSMetrics
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
    validate_court_coordinate_normalization,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


_TRAJECTORY_HEADLINES = (
    "trajectory_rmse_m",
    "trajectory_mean_m",
    "trajectory_p95_m",
    "velocity_head_rmse_mps",
    "position_difference_velocity_rmse_mps",
    "position_difference_acceleration_rmse_mps2",
    "reprojection_rmse_px",
    "below_ground_rate",
    "reprojection_evaluable_rate",
    "clip_trajectory_rmse_mean_m",
)
_EVALUATION_HEADLINES = _TRAJECTORY_HEADLINES + (
    "rollout_position_rmse_m",
    "rollout_velocity_rmse_mps",
)
BLCS_TRAJECTORY_METRIC_CONTRACT = MetricLoggingContract(
    name="BLCS physics trajectory v1",
    stages=(
        StageMetricContract("train", _TRAJECTORY_HEADLINES, ("trajectory_rmse_m",)),
        StageMetricContract("val", _EVALUATION_HEADLINES, ("trajectory_rmse_m",)),
        StageMetricContract("test", _EVALUATION_HEADLINES, ("trajectory_rmse_m",)),
    ),
)


class BLCSLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for BLCS models.

    This module supports both single-view and multiview BLCS training.
    """

    train_metrics: BLCSMetrics
    val_metrics: BLCSMetrics
    test_metrics: BLCSMetrics

    metric_logging_contract = BLCS_TRAJECTORY_METRIC_CONTRACT

    def __init__(
        self,
        config: DictConfig,
        *,
        model_io: TrajectoryBoundModelIO,
    ) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model, training, and loss parameters.

        """
        super().__init__(config)

        self.model_io = model_io
        self.model = model_io.model
        self.io_adapter = cast("TrajectoryModelIOAdapter", model_io.adapter)
        self.court_keypoint_contract = parse_court_keypoint_contract(self.config)
        self.qualitative_rendering = parse_qualitative_rendering(self.config)

        train_cfg = self.config.training
        loss_cfg = self.config.loss
        self.max_epochs = int(train_cfg.trainer.max_epochs)
        self.loss_fn = BLCSLoss(loss_cfg)
        self._last_gradient_diagnostic_step = -1
        gan_enabled = bool(train_cfg.gan.enabled)
        self._initialize_manual_gan(
            discriminator=build_blcs_discriminator(self.config)
            if gan_enabled
            else None,
        )

        metrics_cfg = self.config.metrics
        for stage in ("train", "val", "test"):
            setattr(
                self,
                f"{stage}_metrics",
                BLCSMetrics(
                    rollout_horizons=tuple(metrics_cfg.rollout_horizons)
                    if stage != "train"
                    else (),
                    event_radius_frames=int(metrics_cfg.event_radius_frames),
                ),
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the exact normalization and CourtKP contracts."""
        super().on_save_checkpoint(checkpoint)
        add_court_coordinate_normalization(checkpoint, artifact="BLCS checkpoint")
        write_model_artifact_court_keypoint_contract(
            checkpoint,
            self.court_keypoint_contract,
            location="BLCS checkpoint",
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject normalization or CourtKP mismatches before weights."""
        super().on_load_checkpoint(checkpoint)
        validate_neural_court_observation_order(
            checkpoint, self.court_keypoint_contract
        )
        validate_court_coordinate_normalization(checkpoint, artifact="BLCS checkpoint")
        validate_model_artifact_court_keypoint_contract(
            checkpoint,
            self.court_keypoint_contract,
            location="BLCS checkpoint",
        )

    def _forward_from_batch(
        self, batch: Mapping[str, object]
    ) -> BLCSTrajectoryPrediction:
        """Delegate a validated model invocation to the bound adapter."""
        prediction: BLCSTrajectoryPrediction = self.model_io.run(batch)
        return replace(
            prediction,
            reference_metadata=blcs_reference_metadata_from_batch(batch),
        )

    def _compute_supervised_result(
        self,
        batch: Mapping[str, object],
        stage: str,
    ) -> dict[str, Any]:
        """Compute forward pass, supervised losses, and metrics."""
        reference_metadata = blcs_reference_metadata_from_batch(batch)
        prepared = self.io_adapter.build_training_batch(batch)
        prepared = replace(prepared, reference_metadata=reference_metadata)
        outputs = self.model_io.decode_output(self.model_io.execute_call(prepared.call))
        outputs = replace(outputs, reference_metadata=reference_metadata)

        losses = self.loss_fn(outputs, prepared, step=int(self.global_step))
        if (
            stage == "train"
            and self.global_step % 200 == 0
            and self._last_gradient_diagnostic_step != self.global_step
        ):
            self._log_output_gradients(losses, outputs)
            self._last_gradient_diagnostic_step = int(self.global_step)
        metrics = self._metric_tracker_for_stage(stage).update(
            outputs,
            prepared,
            observed=cast(Tensor, batch["ball_vis"]),
            scene_ids=cast("tuple[str, ...]", batch["scene_id"]),
        )

        return {
            "loss": losses["total"],
            "losses": losses,
            "metrics": {
                **metrics,
                **loss_component_metrics(losses),
            },
            "outputs": outputs,
            "mask": prepared.loss_mask,
            "prepared": prepared,
            "gan_fake": outputs.position,
            "gan_real": prepared.position,
            "gan_padding_mask": ~prepared.loss_mask,
        }

    def _log_output_gradients(
        self, losses: dict[str, Tensor], outputs: BLCSTrajectoryPrediction
    ) -> None:
        """Sample raw and weighted dL/d(normalized output), without changing grads."""
        tensors = {
            key: value
            for key, value in {
                "position": outputs.position,
                "velocity": outputs.velocity,
                "parameters": outputs.physics_parameters,
                "events": outputs.event_logits,
            }.items()
            if value is not None and value.requires_grad
        }
        metrics: dict[str, float] = {}
        for term, weight in self.loss_fn.weights.items():
            if not weight or not tensors:
                continue
            grads = torch.autograd.grad(
                losses[term],
                tuple(tensors.values()),
                retain_graph=True,
                allow_unused=True,
            )
            factor = (
                self.loss_fn.physics_factor(int(self.global_step))
                if term in PHYSICAL_TERMS
                else 1.0
            )
            for name, gradient in zip(tensors, grads, strict=True):
                norm = (
                    outputs.position.new_zeros(())
                    if gradient is None
                    else gradient.detach().float().norm()
                )
                metrics[f"gradient/{term}/{name}_raw_l2"] = float(norm)
                metrics[f"gradient/{term}/{name}_weighted_l2"] = (
                    float(norm) * weight * factor
                )
        # Explicit optimizer-step timestamps, independent of the trainer's
        # scalar logging interval and accumulation boundary.
        if self.logger is not None and self.trainer.is_global_zero:
            self.logger.log_metrics(metrics, step=int(self.global_step))

    def test_prediction_payload(
        self, batch: Mapping[str, object], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Ball 3D position predictions + targets to persist for the test split."""
        outputs = cast("BLCSTrajectoryPrediction", result["outputs"])
        target = self.io_adapter.build_training_batch(batch)
        payload: dict[str, Any] = {
            "pred_position": outputs.position,
            "target_position": target.position,
            "target_velocity": target.velocity,
            "position_3d": target.position,
            "velocity_3d": target.velocity,
            "mask": target.loss_mask,
            "physics_spin": target.physics.spin,
            "physics_segment": target.physics.segment,
            "physics_event": target.physics.event,
            "physics_field": target.physics.field,
            "physics_clock": target.physics.clock,
            "physics_surface": target.physics.surface,
            "frame_index": target.physics.frame_index,
            "clip_frames": target.physics.clip_frames,
            "bounce_count": target.physics.bounce_count,
        }
        if outputs.velocity is not None:
            payload["pred_velocity"] = outputs.velocity
        if outputs.physics_parameters is not None:
            payload["physics_parameters"] = outputs.physics_parameters
            payload["event_logits"] = outputs.event_logits
        # Fixed artifact dimensions make padding independent of batch grouping.
        time_keys = {
            "pred_position",
            "target_position",
            "target_velocity",
            "position_3d",
            "velocity_3d",
            "mask",
            "physics_spin",
            "physics_segment",
            "physics_event",
            "frame_index",
            "pred_velocity",
            "physics_parameters",
            "event_logits",
        }
        width = int(self.config.data.seq_len_range[1])
        for key in time_keys & payload.keys():
            value = payload[key]
            fill = -1 if key in {"physics_segment", "frame_index"} else 0
            padded = value.new_full((value.shape[0], width, *value.shape[2:]), fill)
            padded[:, : value.shape[1]] = value
            payload[key] = padded
        for key in (
            "ball_uv",
            "ball_vis",
            "ball_uv_target",
            "ball_vis_target",
            "court_kp",
            "court_vis",
            "padding_mask",
            "camera_R",
            "camera_C",
            "camera_f",
            "camera_cx",
            "camera_cy",
            "camera_w",
            "camera_h",
        ):
            value = cast(Tensor, batch[key])
            shape = [
                value.shape[0],
                int(self.config.model.max_num_cameras),
                *value.shape[2:],
            ]
            if key.startswith(("ball_", "court_")) or key == "padding_mask":
                shape[2] = width
            fill = 1 if key in {"padding_mask", "camera_w", "camera_h"} else 0
            padded = value.new_full(shape, fill)
            padded[tuple(slice(0, n) for n in value.shape)] = value
            payload[key] = padded
        if outputs.reference_metadata is not None:
            payload.update(
                outputs.reference_metadata.prediction_payload(
                    max_views=int(self.config.data.num_views_range[1]),
                )
            )
        return payload

    def save_test_predictions(
        self,
        metrics: dict[str, Any] | None = None,
        diagnostic_metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        def finite(values: dict[str, Any] | None) -> dict[str, Any] | None:
            return (
                None
                if values is None
                else {
                    key: value if math.isfinite(float(value)) else None
                    for key, value in values.items()
                }
            )

        saved = super().save_test_predictions(
            finite(metrics), finite(diagnostic_metrics)
        )
        if saved is not None:
            report = (
                json.dumps(self.test_metrics.report(), indent=2, allow_nan=False) + "\n"
            )
            for directory in {saved.parent, self._run_predictions_dir()}:
                (directory / "evaluation.json").write_text(report)
        return cast("Path | None", saved)

    def _log_stage_metrics(
        self, stage: str, loss: Tensor, metrics: dict[str, Any]
    ) -> None:
        self.metric_logging_contract.for_stage(stage)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "test",
        )
        for name, value in metrics.items():
            if name.startswith("loss_"):
                self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True)

    def _metric_tracker_for_stage(self, stage: str) -> BLCSMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        if stage == "test":
            return self.test_metrics
        raise ValueError(f"Unknown BLCS metric stage: {stage!r}.")

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

            anim = renderer.create_comparison_animation(
                gt,
                pred,
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
