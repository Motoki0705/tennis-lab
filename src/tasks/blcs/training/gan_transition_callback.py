"""Callback that switches BLCS training from supervised to hybrid GAN mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import pytorch_lightning as pl
from torch import Tensor

if TYPE_CHECKING:
    from omegaconf import DictConfig


class GANTransitionCallback(pl.Callback):
    """Detect supervised convergence and enable hybrid GAN training."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        gan_cfg = (config.get("training", {}) or {}).get("gan", {}) or {}
        transition_cfg = gan_cfg.get("transition", {}) or {}

        self.enabled = bool(gan_cfg.get("enabled", False))
        self.monitor = str(transition_cfg.get("monitor", "val/pos_error_m"))
        self.mode = str(transition_cfg.get("mode", "min"))
        self.min_delta = float(transition_cfg.get("min_delta", 1.0e-3))
        self.patience = int(transition_cfg.get("patience", 3))
        self.min_supervised_epochs = int(transition_cfg.get("min_supervised_epochs", 1))
        self.gan_target_weight = float(gan_cfg.get("target_weight", 0.1))
        self.gan_warmup_epochs = int(gan_cfg.get("warmup_epochs", 5))

        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unsupported transition mode '{self.mode}'. Use ['min', 'max'].")

        self.best_score: float | None = None
        self.bad_epochs = 0
        self.has_switched_to_gan = False
        self.switch_epoch: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
            "has_switched_to_gan": self.has_switched_to_gan,
            "switch_epoch": self.switch_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_score = state_dict.get("best_score")
        self.bad_epochs = int(state_dict.get("bad_epochs", 0))
        self.has_switched_to_gan = bool(state_dict.get("has_switched_to_gan", False))
        self.switch_epoch = state_dict.get("switch_epoch")

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.set_gan_weight(0.0)

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.enabled or not self.has_switched_to_gan or self.switch_epoch is None:
            return

        epochs_since_switch = max(trainer.current_epoch - self.switch_epoch + 1, 0)
        if self.gan_warmup_epochs <= 1:
            gan_weight = self.gan_target_weight
        else:
            progress = min(epochs_since_switch / self.gan_warmup_epochs, 1.0)
            gan_weight = self.gan_target_weight * progress
        pl_module.set_gan_weight(gan_weight)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.enabled or self.has_switched_to_gan:
            return

        if trainer.current_epoch + 1 < self.min_supervised_epochs:
            return

        monitor_value = trainer.callback_metrics.get(self.monitor)
        if monitor_value is None:
            return

        current = float(cast(Tensor, monitor_value).detach().cpu().item())
        if self.patience <= 0:
            self._switch_to_gan(trainer, pl_module)
            return

        if self.best_score is None or self._is_improvement(current):
            self.best_score = current
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self._switch_to_gan(trainer, pl_module)

    def _is_improvement(self, current: float) -> bool:
        assert self.best_score is not None
        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        return current > (self.best_score + self.min_delta)

    def _switch_to_gan(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.has_switched_to_gan = True
        self.switch_epoch = trainer.current_epoch + 1
        pl_module.activate_gan_phase(self.switch_epoch)
        pl_module.set_gan_weight(0.0)