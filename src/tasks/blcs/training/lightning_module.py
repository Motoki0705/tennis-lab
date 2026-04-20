"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.tasks.blcs.data.types import BLCSBatch, BLCSMultiViewBatch
from src.tasks.blcs.models import build_blcs_discriminator, build_blcs_model
from src.tasks.blcs.training.gan_loss import LSGANLoss
from src.tasks.blcs.training.gan_training_strategy import BLCSGANTrainingStrategy
from src.tasks.blcs.training.losses import BLCSLoss
from src.tasks.blcs.training.metrics import BLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSLightningModule(BaseLightningModule):
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

        gan_cfg = train_cfg.get("gan", {}) or {}
        self.gan_enabled = bool(gan_cfg.get("enabled", False))
        self.automatic_optimization = not self.gan_enabled
        scheduler_interval = "epoch" if self.warmup_epochs is not None else "step"
        self.gan_training = (
            BLCSGANTrainingStrategy(
                generator_gradient_clip_val=gan_cfg.get("generator_gradient_clip_val"),
                discriminator_gradient_clip_val=gan_cfg.get("discriminator_gradient_clip_val"),
                scheduler_interval=scheduler_interval,
            )
            if self.gan_enabled
            else None
        )
        self.discriminator = build_blcs_discriminator(self.config) if self.gan_enabled else None
        self.gan_loss_fn = LSGANLoss() if self.gan_enabled else None

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
        ball_mask = batch.get("ball_mask")
        if ball_mask is None:
            return None
        if ball_mask.ndim == 2:
            return ball_mask
        if ball_mask.ndim == 3:
            return (ball_mask > 0).any(dim=1)
        raise ValueError(
            f"ball_mask must have 2 or 3 dims, got shape {tuple(ball_mask.shape)}"
        )

    def _select_metrics(self, stage: str) -> BLCSMetrics:
        """Return metrics object for the current stage."""
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def activate_gan_phase(self, start_epoch: int) -> None:
        """Enable hybrid GAN training from the given epoch onward."""
        if self.gan_training is None:
            return
        self.reset_gan_phase_schedules(start_epoch)
        self.gan_training.activate_phase(start_epoch)

    def set_gan_weight(self, weight: float) -> None:
        """Set the current adversarial loss weight."""
        if self.gan_training is None:
            return
        self.gan_training.set_weight(weight)

    @property
    def gan_phase_active(self) -> bool:
        """Expose current GAN phase state for callbacks and tests."""
        return self.gan_training.phase_active if self.gan_training is not None else False

    @property
    def current_gan_weight(self) -> float:
        """Expose current adversarial weight for logging and tests."""
        return self.gan_training.current_weight if self.gan_training is not None else 0.0

    @property
    def supervised_only_step_count(self) -> int:
        """Expose supervised-only update count for tests."""
        return self.gan_training.supervised_only_step_count if self.gan_training is not None else 0

    @property
    def hybrid_gan_step_count(self) -> int:
        """Expose hybrid GAN update count for tests."""
        return self.gan_training.hybrid_gan_step_count if self.gan_training is not None else 0

    def _unwrap_optimizer(self, optimizer: Any) -> Any:
        """Return the bare optimizer when Lightning wraps it."""
        return optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer

    def _manual_optimizers(self) -> tuple[Any, Any]:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)) or len(optimizers) != 2:
            raise RuntimeError("Expected generator and discriminator optimizers in manual mode.")
        return optimizers[0], optimizers[1]

    def _manual_schedulers(self) -> list[Any]:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return []
        if isinstance(schedulers, (list, tuple)):
            return list(schedulers)
        return [schedulers]

    def _build_optimizer_for_parameters(self, parameters: Any) -> AdamW:
        kwargs: dict[str, Any] = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        if self.optimizer_betas is not None:
            kwargs["betas"] = self.optimizer_betas
        return AdamW(parameters, **kwargs)

    def _build_scheduler_for_optimizer(
        self,
        optimizer: AdamW,
        *,
        total_steps_override: int | None = None,
        max_epochs_override: int | None = None,
    ) -> Any:
        if self.warmup_epochs is not None:
            warmup_epochs = int(self.warmup_epochs)
            max_epochs = (
                int(max_epochs_override) if max_epochs_override is not None else int(self.max_epochs)
            )
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(max_epochs - warmup_epochs, 1),
                    eta_min=self.min_lr,
                )
                return SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            return CosineAnnealingLR(
                optimizer,
                T_max=max(max_epochs, 1),
                eta_min=self.min_lr,
            )

        warmup_steps = int(self.warmup_steps or 0)
        total_steps = (
            int(total_steps_override)
            if total_steps_override is not None
            else self._estimate_total_steps()
        )
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=self.min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        return CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=self.min_lr,
        )

    def reset_gan_phase_schedules(self, start_epoch: int) -> None:
        """Reset generator/discriminator LR schedules when GAN training starts."""
        if not self.gan_enabled:
            return

        schedulers = self._manual_schedulers()
        if len(schedulers) < 2:
            return

        generator_optimizer, discriminator_optimizer = self._manual_optimizers()
        optimizers = [generator_optimizer, discriminator_optimizer]
        remaining_total_steps = max(self._estimate_total_steps() - int(self.global_step), 1)
        remaining_epochs = max(int(self.max_epochs) - int(start_epoch), 1)

        for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
            fresh_optimizer = self._build_optimizer_for_parameters([torch.nn.Parameter(torch.zeros(()))])
            fresh_scheduler = self._build_scheduler_for_optimizer(
                fresh_optimizer,
                total_steps_override=remaining_total_steps,
                max_epochs_override=remaining_epochs,
            )
            scheduler.load_state_dict(fresh_scheduler.state_dict())

            current_groups = self._unwrap_optimizer(optimizer).param_groups
            fresh_groups = fresh_optimizer.param_groups
            for current_group, fresh_group in zip(current_groups, fresh_groups, strict=True):
                current_group["lr"] = fresh_group["lr"]
                if "initial_lr" in fresh_group:
                    current_group["initial_lr"] = fresh_group["initial_lr"]

    def configure_optimizers(self) -> Any:
        """Configure optional generator/discriminator optimizers for GAN mode."""
        if not self.gan_enabled:
            return super().configure_optimizers()
        if self.discriminator is None:
            raise RuntimeError("Discriminator must be instantiated when GAN is enabled.")

        generator_optimizer = self._build_optimizer_for_parameters(self.model.parameters())
        discriminator_optimizer = self._build_optimizer_for_parameters(
            self.discriminator.parameters()
        )
        generator_scheduler = self._build_scheduler_for_optimizer(generator_optimizer)
        discriminator_scheduler = self._build_scheduler_for_optimizer(discriminator_optimizer)
        return [generator_optimizer, discriminator_optimizer], [
            generator_scheduler,
            discriminator_scheduler,
        ]

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
            target_uv=batch.get("ball_uv"),
            target_vis=batch.get("ball_vis"),
            camera_R=batch.get("camera_R"),
            camera_C=batch.get("camera_C"),
            camera_f=batch.get("camera_f"),
            camera_cx=batch.get("camera_cx"),
            camera_cy=batch.get("camera_cy"),
            camera_w=batch.get("camera_w"),
            camera_h=batch.get("camera_h"),
        )

        metrics = self._select_metrics(stage).update(
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
        }

    def training_step(self, batch: BLCSBatch | BLCSMultiViewBatch, batch_idx: int) -> Tensor:
        """Training step."""
        _ = batch_idx
        if self.gan_training is not None:
            loss, metrics = self.gan_training.shared_step(self, batch, "train")
        else:
            result = self._compute_supervised_result(batch, "train")
            loss, metrics = result["loss"], result["metrics"]
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_error_m", metrics.get("position_error_m", 0), prog_bar=True)
        if self.gan_enabled:
            self.log("train/gan_weight", float(self.current_gan_weight))
            self.log("train/gan_phase_active", float(self.gan_phase_active))
            if "loss_gan_generator" in metrics:
                self.log("train/loss_gan_generator", metrics["loss_gan_generator"])
            if "loss_gan_discriminator" in metrics:
                self.log("train/loss_gan_discriminator", metrics["loss_gan_discriminator"])
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch."""
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()
        if self.gan_training is not None:
            self.gan_training.on_train_epoch_end(self)

    def validation_step(self, batch: BLCSBatch | BLCSMultiViewBatch, batch_idx: int) -> None:
        """Validation step."""
        _ = batch_idx
        if self.gan_training is not None:
            loss, metrics = self.gan_training.shared_step(self, batch, "val")
        else:
            result = self._compute_supervised_result(batch, "val")
            loss, metrics = result["loss"], result["metrics"]
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos_error_m", metrics.get("position_error_m", 0), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Called at end of validation epoch."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: BLCSBatch | BLCSMultiViewBatch, batch_idx: int) -> None:
        """Test step."""
        _ = batch_idx
        if self.gan_training is not None:
            loss, metrics = self.gan_training.shared_step(self, batch, "test")
        else:
            result = self._compute_supervised_result(batch, "test")
            loss, metrics = result["loss"], result["metrics"]
        self.log("test/loss", loss)
        self.log("test/pos_error_m", metrics.get("position_error_m", 0))

    def on_test_epoch_end(self) -> None:
        """Called at end of test epoch."""
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
