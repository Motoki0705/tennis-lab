"""Callback that switches training from supervised to hybrid GAN mode.

The transition is *deterministic*: the GAN phase activates at a fixed,
pre-scheduled epoch (``training.gan.transition.start_epoch``) rather than being
triggered by loss-plateau detection. For a 200-epoch run with
``start_epoch: 100``, epochs 0..99 are pure supervised and the adversarial
phase begins at epoch 100 (0-based ``trainer.current_epoch``, i.e. after 100
supervised epochs), ramping the GAN weight up over ``warmup_epochs``.

Choosing epoch monitoring over loss monitoring keeps the schedule reproducible
and independent of the noisy per-run convergence curve, which matters when the
same recipe is re-run across the physics-prior / architecture experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as pl

if TYPE_CHECKING:
    from omegaconf import DictConfig


class GANTransitionCallback(pl.Callback):
    """Activate hybrid GAN training at a fixed, pre-scheduled epoch."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        gan_cfg = (config.get("training", {}) or {}).get("gan", {}) or {}
        transition_cfg = gan_cfg.get("transition", {}) or {}

        self.enabled = bool(gan_cfg.get("enabled", False))

        start_epoch = transition_cfg.get("start_epoch")
        if start_epoch is None:
            if self.enabled:
                # A GAN run with no transition schedule is a config error, not a
                # silent "never switch" no-op (AGENTS.md: no quiet fallbacks).
                raise ValueError(
                    "training.gan.transition.start_epoch is required when GAN is "
                    "enabled (deterministic epoch-based transition)."
                )
            start_epoch = 0
        self.start_epoch = int(start_epoch)
        if self.start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {self.start_epoch}.")

        self.gan_target_weight = float(gan_cfg.get("target_weight", 0.1))
        self.gan_warmup_epochs = int(gan_cfg.get("warmup_epochs", 5))

        # Process-local guard so the one-time phase activation (which rebuilds the
        # optimizer/scheduler state) runs exactly once. Intentionally NOT persisted
        # in state_dict: on resume the module is reconstructed with the GAN phase
        # inactive, so a fresh process must re-activate it deterministically from
        # ``current_epoch >= start_epoch``.
        self.has_switched_to_gan = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.set_gan_weight(0.0)
        if not self.enabled:
            return

        max_epochs = getattr(trainer, "max_epochs", None)
        if max_epochs is not None and self.start_epoch >= int(max_epochs):
            raise ValueError(
                f"training.gan.transition.start_epoch ({self.start_epoch}) must be "
                f"< trainer.max_epochs ({max_epochs}); otherwise the GAN phase "
                "never activates."
            )

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.enabled or trainer.current_epoch < self.start_epoch:
            return

        if not self.has_switched_to_gan:
            self.has_switched_to_gan = True
            # Pass the *actual* current epoch (== start_epoch on the normal
            # transition, > start_epoch when resuming into the GAN phase) so the
            # GAN-phase LR schedule is scaled over the true remaining epochs.
            pl_module.activate_gan_phase(trainer.current_epoch)

        pl_module.set_gan_weight(self._ramp_weight(trainer.current_epoch))

    def _ramp_weight(self, current_epoch: int) -> float:
        # Warmup is anchored to ``start_epoch`` (not the activation epoch) so a
        # resumed run picks the correct point on the ramp for its epoch.
        epochs_since_switch = max(current_epoch - self.start_epoch + 1, 0)
        if self.gan_warmup_epochs <= 1:
            return self.gan_target_weight
        progress = min(epochs_since_switch / self.gan_warmup_epochs, 1.0)
        return self.gan_target_weight * progress
