"""Unified PyTorch Lightning module for PLCS training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_animation
from src.tasks.plcs.models import build_plcs_discriminator, build_plcs_model
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.mcmc import LangevinNoiseInjector, MCMCConfig
from src.tasks.plcs.training.metrics import PLCSMetrics
from src.utils.tensor_utils import normalize_padding_mask

# Visualization imports are deferred to render_qualitative_samples to avoid
# a circular import cycle (visualization.api.predict → inference.predictor →
# training.lightning_module → visualization).
# At function call time the full package graph is already initialised.

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PLCSLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for unified PLCS I/O training."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        self.model: nn.Module = build_plcs_model(self.config)
        self.predict_canonical_pose = bool(
            (self.config.get("model", {}) or {}).get("predict_canonical_pose", False)
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

        # MCMC (SGLD) training strategy (issue #519): optional Langevin noise
        # injection to escape the 180deg rotation flat-saddle local optimum.
        mcmc_cfg = MCMCConfig.from_dict(
            dict((self.config.get("training", {}) or {}).get("mcmc", {}) or {})
        )
        self.mcmc_injector = (
            LangevinNoiseInjector(mcmc_cfg) if mcmc_cfg.enabled else None
        )

        gan_enabled = bool(((self.config.get("training", {}) or {}).get("gan", {}) or {}).get("enabled", False))
        self._initialize_manual_gan(
            discriminator=build_plcs_discriminator(self.config) if gan_enabled else None,
        )

        metrics_cfg = self.config.get("metrics", {})

        def _build_metrics() -> PLCSMetrics:
            return PLCSMetrics(
                position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
                angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
            )

        self.train_metrics = _build_metrics()
        self.val_metrics = _build_metrics()
        self.test_metrics = _build_metrics()

    def _pose_sequence(self, position: Tensor, rotation: Tensor) -> Tensor:
        if position.ndim == 2:
            position = position.unsqueeze(1)
            rotation = rotation.unsqueeze(1)
        return torch.cat([position, rotation], dim=-1)

    def _forward_from_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = self.model(
            human_kp=batch["human_kp"],
            court_kp=batch["court_kp"],
            human_vis=batch.get("human_vis"),
            human_mask=batch.get("human_mask"),
            court_vis=batch.get("court_vis"),
        )
        return result

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
        outputs = self._forward_from_batch(batch)
        human_mask = batch.get("human_mask")
        frame_mask = normalize_padding_mask(human_mask, flatten=False)

        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            pred_canonical_pose=outputs.get("canonical_pose"),
            target_human_kp_3d=batch.get("human_kp_3d"),
            human_mask=human_mask,
        )

        # Auxiliary supervision heads for representation learning on task trunks.
        # Each entry maps an output key -> (target batch key, loss kind, weight
        # name). "position" teaches multiview triangulation; "rotation" teaches
        # heading. aux_position is the ex10 ingredient on the rotation trunk;
        # aux_*_canonical are the issue #520 canonical-trunk variants.
        aux_specs = (
            ("aux_position", "position", "position", "position"),
            ("aux_position_canonical", "position", "position", "position"),
            ("aux_rotation_canonical", "rotation", "rotation", "rotation"),
        )
        for out_key, target_key, kind, weight_name in aux_specs:
            if out_key not in outputs:
                continue
            aux_value = self._aux_loss(
                outputs[out_key], batch[target_key], kind, frame_mask
            )
            losses[out_key] = aux_value
            losses["total"] = (
                losses["total"] + self.loss_fn.weight_for(weight_name) * aux_value
            )

        metrics = self._metric_tracker_for_stage(stage).update(
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
        if "loss_canonical_pose" in metrics:
            self.log(f"{stage}/loss_canonical_pose", metrics["loss_canonical_pose"])
        self._log_gan_metrics(stage, metrics)

    def configure_optimizers(self) -> Any:
        return self.configure_gan_optimizers(self.model.parameters())

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
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
        renderer = PLCSSceneRenderer()

        for batch_idx, batch in enumerate(batches):
            batch_dev = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            try:
                with torch.no_grad():
                    out = self._forward_from_batch(batch_dev)

                # Build CPU-side scenes (adapter handles .cpu() internally)
                batch_cpu = {
                    k: v.cpu() if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                out_cpu = {
                    k: v.cpu() if isinstance(v, Tensor) else v
                    for k, v in out.items()
                }
                gt_scene, pred_scene = batch_to_pose_render_scenes(
                    batch_cpu, out_cpu, sample_idx=0
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
                )
                save_qualitative_animation(
                    animation=anim,
                    artifact_dir=artifact_dir,
                    name=f"plcs_batch{batch_idx:02d}",
                    tb_writer=tb_writer,
                    tag=f"qualitative/plcs/batch{batch_idx:02d}",
                    global_step=global_step,
                )
            except Exception:
                logger.exception(
                    "render_qualitative_samples: failed for batch_idx=%d", batch_idx
                )
