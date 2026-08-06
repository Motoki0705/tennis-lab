"""Manual-optimization Lightning module for the staged #579 schedule.

The variable-T batch sampler emits, per optimizer-step group, ``accumulate(T)``
consecutive micro-batches of size ``B(T)`` sharing one ``T``. This module runs
manual optimization: it scales each micro-batch loss by ``accumulate``,
accumulates gradients, and steps the optimizer once per group, so the effective
batch size stays constant across ``T`` even though the physical batch size
varies. The learning-rate scheduler is stepped once per epoch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor

from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class StagedBallDetectionLightningModule(BallDetectionLightningModule):
    """Ball detection with per-group gradient accumulation for variable T."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        if self.gan_enabled:
            raise ValueError("Staged training does not support the GAN path.")
        # Manual optimization: we control backward/step/scheduler ourselves.
        self.automatic_optimization = False

        staged_cfg = self.config.training.staged
        self.effective_batch_size = int(staged_cfg.effective_batch_size)
        # Clip lives under training.staged (NOT training.trainer): Lightning
        # forbids Trainer-level gradient_clip_val under manual optimization, so
        # we clip here via self.clip_gradients instead.
        clip = staged_cfg.gradient_clip_val
        self.gradient_clip_val = None if clip is None else float(clip)

        self._accum_count = 0

    def _clip_and_step(self) -> None:
        optimizer = self.optimizers()
        if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
        optimizer.step()
        optimizer.zero_grad()

    def set_effective_batch_size(self, value: int) -> None:
        """Sync EBS with the datamodule's calibrated plan (called by the runner)."""
        self.effective_batch_size = int(value)

    def on_train_epoch_start(self) -> None:
        # Drop any stray partial-group gradients carried across the epoch break.
        self._accum_count = 0
        self.optimizers().zero_grad()

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        _ = batch_idx
        physical_batch = int(batch["images"].shape[0])
        accumulate = max(1, round(self.effective_batch_size / max(physical_batch, 1)))

        result = self._compute_supervised_result(batch, "train")
        loss = result["loss"]
        self.manual_backward(loss / accumulate)
        self._accum_count += 1

        if self._accum_count >= accumulate:
            self._clip_and_step()
            self._accum_count = 0

        self._log_stage_metrics("train", loss.detach(), result["metrics"])
        self.log("train/physical_batch", float(physical_batch), prog_bar=False)
        self.log("train/seq_len", float(batch["images"].shape[1]), prog_bar=True)
        return loss.detach()

    def on_train_epoch_end(self) -> None:
        # Flush a trailing partial group (defensive; the sampler emits whole
        # groups, so this normally does nothing).
        if self._accum_count > 0:
            self._clip_and_step()
            self._accum_count = 0
        super().on_train_epoch_end()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()


__all__ = ["StagedBallDetectionLightningModule"]
