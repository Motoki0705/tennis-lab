"""GAN training orchestration for BLCS manual optimization."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


class BLCSGANTrainingStrategy:
    """Own the GAN training procedure outside the LightningModule."""

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
        """Enable hybrid GAN training from the given epoch onward."""
        self.phase_active = True
        self.start_epoch = int(start_epoch)

    def set_weight(self, weight: float) -> None:
        """Update the current adversarial loss weight."""
        self.current_weight = max(float(weight), 0.0)

    def shared_step(self, module: Any, batch: Any, stage: str) -> tuple[Tensor, dict[str, float]]:
        """Run the appropriate supervised or GAN training routine."""
        if stage != "train" or not self.phase_active:
            return self._supervised_step(module, batch, stage)
        return self._gan_step(module, batch, stage)

    def on_train_epoch_end(self, module: Any) -> None:
        """Step manual schedulers on epoch boundaries when configured."""
        if self.scheduler_interval != "epoch":
            return

        schedulers = self._manual_schedulers(module)
        if not schedulers:
            return
        schedulers[0].step()
        if self.phase_active and len(schedulers) > 1:
            schedulers[1].step()

    def _unwrap_optimizer(self, optimizer: Any) -> Any:
        """Return the bare optimizer when Lightning wraps it."""
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
    ) -> tuple[Tensor, dict[str, float]]:
        """Run the supervised BLCS training/evaluation step."""
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
    ) -> tuple[Tensor, dict[str, float]]:
        """Run the hybrid supervised + LSGAN training step."""
        if stage != "train":
            return self._supervised_step(module, batch, stage)
        if module.discriminator is None or module.gan_loss_fn is None:
            raise RuntimeError("GAN components are not initialized.")

        result = module._compute_supervised_result(batch, stage)
        generator_optimizer, discriminator_optimizer = self._manual_optimizers(module)

        mask = result["mask"]
        pred_position = result["outputs"]["position"]
        target_position = batch["position_3d"]

        module.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()
        real_logits = module.discriminator(target_position, mask=mask)
        fake_logits = module.discriminator(pred_position.detach(), mask=mask)
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
        fake_logits_for_generator = module.discriminator(pred_position, mask=mask)
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
            "loss_hybrid_total": hybrid_loss.item(),
            "loss_gan_generator": gan_loss.item(),
            "loss_gan_discriminator": discriminator_loss.item(),
            "gan_weight": float(self.current_weight),
            "gan_phase_active": float(self.phase_active),
        }
        return hybrid_loss.detach(), metrics
