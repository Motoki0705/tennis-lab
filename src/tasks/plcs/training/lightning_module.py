"""Unified PyTorch Lightning module for PLCS training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_animation
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import (
    PLCSDecodedPrediction,
    PLCSModelIOAdapter,
    PLCSPreparedBatch,
    PLCSStandardBoundModelIO,
    build_plcs_model_io,
)
from src.tasks.plcs.models.discriminators import build_plcs_discriminator
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.mcmc import LangevinNoiseInjector, MCMCConfig
from src.tasks.plcs.training.metrics import PLCSMetrics
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
    validate_court_coordinate_normalization,
)

# Visualization imports are deferred to render_qualitative_samples to avoid
# a circular import cycle (visualization.api.predict → inference.predictor →
# training.lightning_module → visualization).
# At function call time the full package graph is already initialised.

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PLCSLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for unified PLCS I/O training."""

    def __init__(self, config: DictConfig) -> None:
        runtime = PLCSTrainingConfig.from_config(config)
        super().__init__(config)
        self.plcs_runtime = runtime

        model_io = build_plcs_model_io(runtime)
        adapter = model_io.adapter
        if not isinstance(adapter, PLCSModelIOAdapter):
            raise ValueError(
                "PLCSLightningModule requires a standard PLCS model-I/O pair."
            )
        self.io_adapter = adapter
        self.model_io = cast(PLCSStandardBoundModelIO, model_io)
        self.model: nn.Module = self.model_io.model
        self._add_auxiliary_supervision = (
            self._add_position_auxiliary_supervision
            if self.io_adapter.predict_auxiliary_position
            else self._keep_primary_supervision
        )

        root = runtime.raw
        loss_cfg = PLCSLossConfig.from_dict(dict(root.loss))
        self.loss_fn = PLCSLoss(config=loss_cfg)

        # MCMC (SGLD) training strategy (issue #519): optional Langevin noise
        # injection to escape the 180deg rotation flat-saddle local optimum.
        mcmc_cfg = MCMCConfig.from_dict(dict(root.training.mcmc))
        self.mcmc_injector = (
            LangevinNoiseInjector(mcmc_cfg) if mcmc_cfg.enabled else None
        )

        gan_enabled = runtime.shared.training.gan.enabled
        self._initialize_manual_gan(
            discriminator=build_plcs_discriminator(runtime) if gan_enabled else None,
        )

        metrics_cfg = require_config_mapping(root, "metrics", path="configuration")
        position_threshold = float(
            cast(
                "float | int",
                require_config_value(
                    metrics_cfg, "position_threshold_m", (float, int), path="metrics"
                ),
            )
        )
        angle_threshold = float(
            cast(
                "float | int",
                require_config_value(
                    metrics_cfg, "angle_threshold_deg", (float, int), path="metrics"
                ),
            )
        )

        def _build_metrics() -> PLCSMetrics:
            return PLCSMetrics(
                position_threshold_m=position_threshold,
                angle_threshold_deg=angle_threshold,
            )

        self.train_metrics = _build_metrics()
        self.val_metrics = _build_metrics()
        self.test_metrics = _build_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        add_court_coordinate_normalization(checkpoint, artifact="PLCS checkpoint")

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validate_court_coordinate_normalization(checkpoint, artifact="PLCS checkpoint")

    def _pose_sequence(self, position: Tensor, rotation: Tensor) -> Tensor:
        if position.ndim == 2:
            position = position.unsqueeze(1)
            rotation = rotation.unsqueeze(1)
        return torch.cat([position, rotation], dim=-1)

    def _forward_from_batch(
        self, batch: dict[str, Tensor]
    ) -> tuple[PLCSDecodedPrediction, PLCSPreparedBatch]:
        prepared = self.io_adapter.prepare_training_batch(batch)
        raw_output = self.model_io.execute_call(prepared.call)
        return self.io_adapter.decode_prepared_output(raw_output, prepared), prepared

    def _metric_tracker_for_stage(self, stage: str) -> PLCSMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _aux_loss(
        self,
        pred: Tensor,
        target: Tensor,
        kind: str,
        frame_mask: Tensor | None,
    ) -> Tensor:
        """Masked auxiliary loss for a representation-learning head.

        ``kind="position"`` uses smooth-L1; ``kind="rotation"`` uses the same
        ``1 - cosine`` loss as the main rotation term on the ``(cos, sin)``
        representation.
        """
        if kind == "rotation":
            pred_norm = nn.functional.normalize(pred, dim=-1)
            target_norm = nn.functional.normalize(target, dim=-1)
            per_frame = 1.0 - (pred_norm * target_norm).sum(dim=-1)
        else:
            per_frame = nn.functional.smooth_l1_loss(
                pred, target, reduction="none"
            ).mean(dim=-1)
        if frame_mask is not None and per_frame.shape == frame_mask.shape:
            from src.utils.tensor_utils import masked_mean  # noqa: PLC0415

            return masked_mean(per_frame, frame_mask, binarize=True, denom_min=1.0)
        return per_frame.mean()

    def _compute_supervised_result(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Any]:
        outputs, prepared = self._forward_from_batch(batch)
        target_position = cast(Tensor, prepared.target_position)
        target_rotation = cast(Tensor, prepared.target_rotation)
        padding_mask = prepared.target_padding_mask
        frame_mask = None
        gan_padding_mask = None
        if padding_mask is not None:
            gan_padding_mask = (
                padding_mask.all(dim=1) if padding_mask.ndim == 3 else padding_mask
            )
            frame_mask = ~gan_padding_mask

        loss_inputs = self.loss_fn.prepare_inputs(
            pred_position=outputs.position,
            pred_rotation=outputs.rotation,
            target_position=target_position,
            target_rotation=target_rotation,
            pred_canonical_pose=outputs.canonical_pose,
            target_human_kp_3d=prepared.target_human_kp_3d,
            padding_mask=padding_mask,
        )
        losses = self.loss_fn(loss_inputs)

        losses = self._add_auxiliary_supervision(
            losses,
            outputs,
            target_position,
            frame_mask,
        )

        metrics = self._metric_tracker_for_stage(stage).update(
            outputs.position,
            outputs.rotation,
            target_position,
            target_rotation,
            padding_mask=padding_mask,
        )

        return {
            "loss": losses["total"],
            "metrics": {
                **metrics,
                **{f"loss_{k}": float(v.item()) for k, v in losses.items()},
            },
            "outputs": outputs,
            "prepared": prepared,
            "gan_fake": self._pose_sequence(outputs.position, outputs.rotation),
            "gan_real": self._pose_sequence(target_position, target_rotation),
            "gan_padding_mask": gan_padding_mask,
        }

    @staticmethod
    def _keep_primary_supervision(
        losses: dict[str, Tensor],
        outputs: PLCSDecodedPrediction,
        target_position: Tensor,
        frame_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        del outputs, target_position, frame_mask
        return losses

    def _add_position_auxiliary_supervision(
        self,
        losses: dict[str, Tensor],
        outputs: PLCSDecodedPrediction,
        target_position: Tensor,
        frame_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        auxiliary_position = cast(Tensor, outputs.auxiliary_position)
        aux_value = self._aux_loss(
            auxiliary_position,
            target_position,
            "position",
            frame_mask,
        )
        losses["aux_position"] = aux_value
        losses["total"] = (
            losses["total"] + self.loss_fn.weight_for("position") * aux_value
        )
        return losses

    def _log_stage_metrics(
        self, stage: str, loss: Tensor, metrics: dict[str, Any]
    ) -> None:
        prog_bar = stage != "test"
        self.log(f"{stage}/loss", loss, prog_bar=prog_bar)
        self.log(
            f"{stage}/pos_error_m",
            metrics.get("position_error_m", 0.0),
            prog_bar=prog_bar,
        )
        self.log(
            f"{stage}/ang_error_deg",
            metrics.get("angular_error_deg", 0.0),
            prog_bar=prog_bar,
        )
        if "loss_canonical_pose" in metrics:
            self.log(f"{stage}/loss_canonical_pose", metrics["loss_canonical_pose"])
        self._log_gan_metrics(stage, metrics)

    def test_prediction_payload(
        self, batch: dict[str, Tensor], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Position/rotation predictions + targets to persist for the test split."""
        outputs = cast(PLCSDecodedPrediction, result["outputs"])
        prepared = cast(PLCSPreparedBatch, result["prepared"])
        payload: dict[str, Any] = {
            "pred_position": outputs.position,
            "pred_rotation": outputs.rotation,
            "target_position": prepared.target_position,
            "target_rotation": prepared.target_rotation,
        }
        mask = result.get("gan_padding_mask")
        if mask is not None:
            payload["padding_mask"] = mask
        return payload

    def configure_optimizers(self) -> Any:
        return self.configure_gan_optimizers(self.model.parameters())

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Inject SGLD/Langevin noise after the optimizer step (issue #519).

        Runs in both automatic (baseline) and manual (GAN) optimization modes
        because Lightning calls this hook after the full training step. The
        generator optimizer's current LR sets the Langevin noise scale.
        """
        if self.mcmc_injector is None:
            return
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        lr = float(optimizer.param_groups[0]["lr"])
        total_steps = max(self._estimate_total_steps(), 1)
        progress = float(self.global_step) / float(total_steps)
        std = self.mcmc_injector.inject(
            self.model,
            lr=lr,
            epoch=int(self.current_epoch),
            progress=progress,
        )
        self.log("train/mcmc_noise_std", std)

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
        """Render GT vs predicted player positions using PLCSSceneRenderer.

        For each batch, builds GT and Pred :class:`PoseRenderScene` objects via
        the adapter, creates a comparison animation with
        :meth:`PLCSSceneRenderer.create_comparison_animation`, and persists the
        result with :func:`save_qualitative_animation`.

        The view is selected automatically:
        - ``"3d"`` when both scenes have ``canonical_pose_3d`` (requires the
          model to output ``canonical_pose`` and ``human_kp_3d`` in the batch).
        - ``"2d_topdown"`` otherwise (position/rotation only).
        """
        # Deferred imports to break circular dependency at module load time.
        from src.tasks.plcs.visualization.adapters.render_inputs import (  # noqa: PLC0415
            batch_to_pose_render_scenes,
        )
        from src.tasks.plcs.visualization.rendering.scene_renderer import (  # noqa: PLC0415
            PLCSSceneRenderer,
        )

        device = next(self.parameters()).device
        renderer = PLCSSceneRenderer(
            style=self.plcs_runtime.qualitative_style,
            camera=self.plcs_runtime.qualitative_view_3d,
        )

        for batch_idx, batch in enumerate(batches):
            batch_dev = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            try:
                with torch.no_grad():
                    out, _ = self._forward_from_batch(batch_dev)

                # Build CPU-side scenes (adapter handles .cpu() internally)
                batch_cpu = {
                    k: v.cpu() if isinstance(v, Tensor) else v for k, v in batch.items()
                }
                gt_scene, pred_scene = batch_to_pose_render_scenes(
                    batch_cpu,
                    PLCSDecodedPrediction(
                        position=out.position.cpu(),
                        rotation=out.rotation.cpu(),
                        canonical_pose=(
                            out.canonical_pose.cpu()
                            if out.canonical_pose is not None
                            else None
                        ),
                        auxiliary_position=(
                            out.auxiliary_position.cpu()
                            if out.auxiliary_position is not None
                            else None
                        ),
                    ),
                    sample_idx=0,
                )

                # Choose view: 3d only when both scenes have canonical pose data
                if (
                    gt_scene.canonical_pose_3d is not None
                    and pred_scene.canonical_pose_3d is not None
                ):
                    view = "3d"
                else:
                    view = "2d_topdown"

                anim = renderer.create_comparison_animation(
                    gt_scene,
                    pred_scene,
                    view=view,
                    fps=self.plcs_runtime.qualitative_fps,
                )
                save_qualitative_animation(
                    animation=anim,
                    artifact_dir=artifact_dir,
                    name=f"plcs_batch{batch_idx:02d}",
                    tb_writer=tb_writer,
                    tag=f"qualitative/plcs/batch{batch_idx:02d}",
                    global_step=global_step,
                    fps=self.plcs_runtime.qualitative_fps,
                )
            except Exception:
                logger.exception(
                    "render_qualitative_samples: failed for batch_idx=%d", batch_idx
                )
