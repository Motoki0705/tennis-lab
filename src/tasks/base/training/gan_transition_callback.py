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

from typing import Any, Protocol, runtime_checkable

import pytorch_lightning as pl

from src.tasks.base.configuration import (
    BaseTrainingConfig,
    as_config_mapping,
    require_config_mapping,
)


@runtime_checkable
class _GANPhaseModule(Protocol):
    def set_gan_weight(self, weight: float) -> None: ...

    def activate_gan_phase(self, current_epoch: int) -> None: ...


def _require_gan_phase_module(module: pl.LightningModule) -> _GANPhaseModule:
    if not isinstance(module, _GANPhaseModule):
        raise TypeError(
            "GANTransitionCallback requires set_gan_weight() and activate_gan_phase()."
        )
    return module


class GANTransitionCallback(pl.Callback):
    """Activate hybrid GAN training at a fixed, pre-scheduled epoch."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        root = as_config_mapping(config, path="configuration")
        training = BaseTrainingConfig.from_validated_task_mapping(
            require_config_mapping(root, "training", path="configuration")
        )
        gan = training.gan
        self.enabled: bool = gan.enabled
        self.start_epoch: int = gan.transition.start_epoch
        self.gan_target_weight: float = gan.target_weight
        self.gan_warmup_epochs: int = gan.warmup_epochs

        # Process-local guard so the one-time phase activation (which rebuilds the
        # optimizer/scheduler state) runs exactly once. Intentionally NOT persisted
        # in state_dict: on resume the module is reconstructed with the GAN phase
        # inactive, so a fresh process must re-activate it deterministically from
        # ``current_epoch >= start_epoch``.
        self.has_switched_to_gan = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        gan_module = _require_gan_phase_module(pl_module)
        gan_module.set_gan_weight(0.0)
        if not self.enabled:
            return

        max_epochs = trainer.max_epochs
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
            _require_gan_phase_module(pl_module).activate_gan_phase(
                trainer.current_epoch
            )

        _require_gan_phase_module(pl_module).set_gan_weight(
            self._ramp_weight(trainer.current_epoch)
        )

    def _ramp_weight(self, current_epoch: int) -> float:
        # Warmup is anchored to ``start_epoch`` (not the activation epoch) so a
        # resumed run picks the correct point on the ramp for its epoch.
        epochs_since_switch = max(current_epoch - self.start_epoch + 1, 0)
        if self.gan_warmup_epochs <= 1:
            return self.gan_target_weight
        progress = min(epochs_since_switch / self.gan_warmup_epochs, 1.0)
        return self.gan_target_weight * progress
