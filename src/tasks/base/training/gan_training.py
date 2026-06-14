"""Shared manual GAN helpers for scene-level training tasks."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.tasks.base.training.gan_loss import LSGANLoss


class ManualGANTrainingStrategy:
    """Own the manual GAN optimization procedure outside the LightningModule."""

    def __init__(
        self,
        *,
        generator_gradient_clip_val: float | None,
        discriminator_gradient_clip_val: float | None,
        scheduler_interval: str,
    ) -> None:
        self.generator_gradient_clip_val = generator_gradient_clip_val
        self.discriminator_gradient_clip_val = discriminator_gradient_clip_val
        self.scheduler_interval = scheduler_interval

        self.phase_active = False
        self.start_epoch: int | None = None
        self.current_weight = 0.0
        self.supervised_only_step_count = 0
        self.hybrid_gan_step_count = 0

    def activate_phase(self, start_epoch: int) -> None:
        self.phase_active = True
        self.start_epoch = int(start_epoch)

    def set_weight(self, weight: float) -> None:
        self.current_weight = max(float(weight), 0.0)

    def shared_step(self, module: Any, batch: Any, stage: str) -> tuple[Tensor, dict[str, Any]]:
        if stage != "train" or not self.phase_active:
            return self._supervised_step(module, batch, stage)
        return self._gan_step(module, batch, stage)

    def on_train_epoch_end(self, module: Any) -> None:
        if self.scheduler_interval != "epoch":
            return

        schedulers = self._manual_schedulers(module)
        if not schedulers:
            return
        schedulers[0].step()
        if self.phase_active and len(schedulers) > 1:
            schedulers[1].step()

    def _unwrap_optimizer(self, optimizer: Any) -> Any:
        return optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer

    def _manual_optimizers(self, module: Any) -> tuple[Any, Any]:
        optimizers = module.optimizers()
        if not isinstance(optimizers, (list, tuple)) or len(optimizers) != 2:
            raise RuntimeError("Expected generator and discriminator optimizers in manual mode.")
        return optimizers[0], optimizers[1]

    def _manual_schedulers(self, module: Any) -> list[Any]:
        schedulers = module.lr_schedulers()
        if schedulers is None:
            return []
        if isinstance(schedulers, (list, tuple)):
            return list(schedulers)
        return [schedulers]

    def _clip_manual_gradients(self, module: Any, optimizer: Any, clip_val: float | None) -> None:
        if clip_val is None or float(clip_val) <= 0:
            return
        clip_algorithm = str(getattr(module.trainer, "gradient_clip_algorithm", "norm")).lower()
        parameters = self._unwrap_optimizer(optimizer).param_groups
        gradients = [
            param
            for group in parameters
            for param in group["params"]
            if param.grad is not None
        ]
        if not gradients:
            return
        if clip_algorithm == "value":
            clip_grad_value_(gradients, float(clip_val))
            return
        clip_grad_norm_(gradients, float(clip_val))

    def _step_manual_schedulers(self, module: Any, *, include_discriminator: bool) -> None:
        if self.scheduler_interval != "step":
            return
        schedulers = self._manual_schedulers(module)
        if not schedulers:
            return
        schedulers[0].step()
        if include_discriminator and len(schedulers) > 1:
            schedulers[1].step()

    def _supervised_step(
        self,
        module: Any,
        batch: Any,
        stage: str,
    ) -> tuple[Tensor, dict[str, Any]]:
        result = module._compute_supervised_result(batch, stage)

        if stage == "train":
            generator_optimizer, _ = self._manual_optimizers(module)
            module.toggle_optimizer(generator_optimizer)
            generator_optimizer.zero_grad()
            module.manual_backward(result["loss"])
            self._clip_manual_gradients(
                module,
                generator_optimizer,
                self.generator_gradient_clip_val,
            )
            generator_optimizer.step()
            module.untoggle_optimizer(generator_optimizer)
            self._step_manual_schedulers(module, include_discriminator=False)
            self.supervised_only_step_count += 1
            return result["loss"].detach(), result["metrics"]

        return result["loss"], result["metrics"]

    def _gan_step(
        self,
        module: Any,
        batch: Any,
        stage: str,
    ) -> tuple[Tensor, dict[str, Any]]:
        if stage != "train":
            return self._supervised_step(module, batch, stage)
        if module.discriminator is None or module.gan_loss_fn is None:
            raise RuntimeError("GAN components are not initialized.")

        result = module._compute_supervised_result(batch, stage)
        generator_optimizer, discriminator_optimizer = self._manual_optimizers(module)

        mask = result.get("gan_mask")
        fake_sequence = result["gan_fake"]
        real_sequence = result["gan_real"]

        module.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()
        real_logits = module.discriminator(real_sequence, mask=mask)
        fake_logits = module.discriminator(fake_sequence.detach(), mask=mask)
        discriminator_loss = module.gan_loss_fn.discriminator_loss(real_logits, fake_logits)
        module.manual_backward(discriminator_loss)
        self._clip_manual_gradients(
            module,
            discriminator_optimizer,
            self.discriminator_gradient_clip_val,
        )
        discriminator_optimizer.step()
        module.untoggle_optimizer(discriminator_optimizer)

        module.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()
        fake_logits_for_generator = module.discriminator(fake_sequence, mask=mask)
        gan_loss = module.gan_loss_fn.generator_loss(fake_logits_for_generator)
        hybrid_loss = result["loss"] + self.current_weight * gan_loss
        module.manual_backward(hybrid_loss)
        self._clip_manual_gradients(
            module,
            generator_optimizer,
            self.generator_gradient_clip_val,
        )
        generator_optimizer.step()
        module.untoggle_optimizer(generator_optimizer)

        self._step_manual_schedulers(module, include_discriminator=True)
        self.hybrid_gan_step_count += 1

        metrics = {
            **result["metrics"],
            "loss_hybrid_total": hybrid_loss.detach(),
            "loss_gan_generator": gan_loss.detach(),
            "loss_gan_discriminator": discriminator_loss.detach(),
            "gan_weight": float(self.current_weight),
            "gan_phase_active": float(self.phase_active),
        }
        return hybrid_loss.detach(), metrics


class ManualGANSupportMixin:
    """Shared manual-optimization plumbing for task-specific GAN modules."""

    gan_enabled: bool
    gan_training: ManualGANTrainingStrategy | None
    discriminator: Any | None
    gan_loss_fn: LSGANLoss | None

    def _initialize_manual_gan(self, *, discriminator: Any | None) -> None:
        train_cfg = self.config.get("training", {}) or {}
        trainer_cfg = train_cfg.get("trainer", {}) or {}
        self.max_epochs = int(trainer_cfg.get("max_epochs", self.max_epochs))

        gan_cfg = train_cfg.get("gan", {}) or {}
        self.gan_enabled = bool(gan_cfg.get("enabled", False))
        self.automatic_optimization = not self.gan_enabled
        scheduler_interval = "epoch" if self.warmup_epochs is not None else "step"
        self.gan_training = (
            ManualGANTrainingStrategy(
                generator_gradient_clip_val=gan_cfg.get("generator_gradient_clip_val"),
                discriminator_gradient_clip_val=gan_cfg.get("discriminator_gradient_clip_val"),
                scheduler_interval=scheduler_interval,
            )
            if self.gan_enabled
            else None
        )
        self.discriminator = discriminator if self.gan_enabled else None
        self.gan_loss_fn = LSGANLoss() if self.gan_enabled else None

    def activate_gan_phase(self, start_epoch: int) -> None:
        if self.gan_training is None:
            return
        self.reset_gan_phase_schedules(start_epoch)
        self.gan_training.activate_phase(start_epoch)

    def set_gan_weight(self, weight: float) -> None:
        if self.gan_training is None:
            return
        self.gan_training.set_weight(weight)

    @property
    def gan_phase_active(self) -> bool:
        return self.gan_training.phase_active if self.gan_training is not None else False

    @property
    def current_gan_weight(self) -> float:
        return self.gan_training.current_weight if self.gan_training is not None else 0.0

    @property
    def supervised_only_step_count(self) -> int:
        return self.gan_training.supervised_only_step_count if self.gan_training is not None else 0

    @property
    def hybrid_gan_step_count(self) -> int:
        return self.gan_training.hybrid_gan_step_count if self.gan_training is not None else 0

    def _shared_stage_step(self, batch: Any, stage: str) -> tuple[Tensor, dict[str, Any]]:
        if self.gan_training is not None:
            return self.gan_training.shared_step(self, batch, stage)
        result = self._compute_supervised_result(batch, stage)
        return result["loss"], result["metrics"]

    def _metric_tracker_for_stage(self, stage: str) -> Any | None:
        _ = stage
        return None

    def _epoch_metric_log_name(self, stage: str, name: str) -> str:
        if stage == "test":
            return f"test/{name}"
        return f"{stage}/epoch_{name}"

    def _log_gan_metrics(self, stage: str, metrics: dict[str, Any]) -> None:
        """Log GAN-specific training metrics.

        Logs ``train/gan_weight`` and ``train/gan_phase_active`` and,
        conditionally, ``train/loss_gan_generator`` /
        ``train/loss_gan_discriminator`` when present in ``metrics``.  Only
        active during the ``train`` stage while GAN training is enabled.
        """
        if stage == "train" and self.gan_enabled:
            self.log("train/gan_weight", float(self.current_gan_weight))
            self.log("train/gan_phase_active", float(self.gan_phase_active))
            if "loss_gan_generator" in metrics:
                self.log("train/loss_gan_generator", metrics["loss_gan_generator"])
            if "loss_gan_discriminator" in metrics:
                self.log("train/loss_gan_discriminator", metrics["loss_gan_discriminator"])

    def _flush_stage_metrics(self, stage: str) -> None:
        tracker = self._metric_tracker_for_stage(stage)
        if tracker is None:
            return

        metrics = tracker.compute()
        for name, value in metrics.items():
            self.log(self._epoch_metric_log_name(stage, name), value)
        tracker.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        _ = batch_idx
        loss, metrics = self._shared_stage_step(batch, "train")
        self._log_stage_metrics("train", loss, metrics)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        _ = batch_idx
        loss, metrics = self._shared_stage_step(batch, "val")
        self._log_stage_metrics("val", loss, metrics)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        _ = batch_idx
        loss, metrics = self._shared_stage_step(batch, "test")
        self._log_stage_metrics("test", loss, metrics)

    def on_train_epoch_end(self) -> None:
        self._flush_stage_metrics("train")
        if self.gan_training is not None:
            self.gan_training.on_train_epoch_end(self)

    def on_validation_epoch_end(self) -> None:
        self._flush_stage_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._flush_stage_metrics("test")

    def _unwrap_optimizer(self, optimizer: Any) -> Any:
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

    def configure_gan_optimizers(self, generator_parameters: Any) -> Any:
        if not self.gan_enabled:
            return super().configure_optimizers()
        if self.discriminator is None:
            raise RuntimeError("Discriminator must be instantiated when GAN is enabled.")

        generator_optimizer = self._build_optimizer_for_parameters(generator_parameters)
        discriminator_optimizer = self._build_optimizer_for_parameters(
            self.discriminator.parameters()
        )
        generator_scheduler = self._build_scheduler_for_optimizer(generator_optimizer)
        discriminator_scheduler = self._build_scheduler_for_optimizer(discriminator_optimizer)
        return [generator_optimizer, discriminator_optimizer], [
            generator_scheduler,
            discriminator_scheduler,
        ]