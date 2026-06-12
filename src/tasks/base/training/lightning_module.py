"""Shared Lightning training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseLightningModule(pl.LightningModule):
    """Base Lightning module with shared optimizer/scheduler logic.

    This class expects training settings under `config.training` and optional
    dataset sizing under `config.data`.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config or {}

        train_cfg = self.config.get("training", {})
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.warmup_steps = train_cfg.get("warmup_steps")
        self.warmup_epochs = train_cfg.get("warmup_epochs")
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.min_lr = train_cfg.get("min_lr", 1e-6)
        optimizer_cfg = train_cfg.get("optimizer", {}) or {}
        betas = optimizer_cfg.get("betas")
        self.optimizer_betas = tuple(betas) if betas is not None else None

    # ------------------------------------------------------------------
    # Qualitative validation logging hook
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
        """Render qualitative validation samples.

        Override in task-specific subclasses to produce visualizations.
        The default implementation is a no-op.

        Args:
            batches: Collected validation batch dicts (CPU tensors).
            outputs: Corresponding validation_step outputs (CPU tensors).
            artifact_dir: Directory to save artifact images/files.
            tb_writer: TensorBoard SummaryWriter (may be ``None``).
            global_step: Current global training step.
            epoch: Current epoch number.
        """

    def _estimate_total_steps(self) -> int:
        steps_per_epoch_attr = getattr(self, "steps_per_epoch", None)
        if steps_per_epoch_attr:
            return int(steps_per_epoch_attr) * int(self.max_epochs)

        estimated_steps = None
        if getattr(self, "_trainer", None) is not None:
            estimated_steps = getattr(self._trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None:
            return int(estimated_steps)

        data_cfg = self.config.get("data", {})
        sim_cfg = self.config.get("simulation", {})
        train_cfg = self.config.get("training", {})
        steps_per_epoch = train_cfg.get("steps_per_epoch")
        if steps_per_epoch is not None:
            return int(steps_per_epoch) * int(self.max_epochs)

        num_samples = data_cfg.get("num_scenes_per_epoch")
        if num_samples is None:
            num_samples = data_cfg.get("num_samples_per_epoch")
        if num_samples is None:
            num_samples = train_cfg.get("num_samples_per_epoch")
        if num_samples is None:
            num_samples = sim_cfg.get("num_train_scenes")
        if num_samples is None:
            num_samples = 10000
        batch_size = data_cfg.get("batch_size", 64)
        steps_per_epoch = max(num_samples // batch_size, 1)
        return steps_per_epoch * self.max_epochs

    def optimizer_param_groups(self) -> list[dict[str, Any]] | None:
        """Optional parameter groups for the optimizer."""
        return None

    def _build_optimizer(self) -> AdamW:
        params = self.optimizer_param_groups()
        if params is None:
            kwargs = {
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
            if self.optimizer_betas is not None:
                kwargs["betas"] = self.optimizer_betas
            return AdamW(self.parameters(), **kwargs)
        for group in params:
            group.setdefault("lr", self.learning_rate)
            group.setdefault("weight_decay", self.weight_decay)
            if self.optimizer_betas is not None:
                group.setdefault("betas", self.optimizer_betas)
        return AdamW(params)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = self._build_optimizer()

        if self.warmup_epochs is not None:
            warmup_epochs = int(self.warmup_epochs)
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(int(self.max_epochs) - warmup_epochs, 1),
                    eta_min=self.min_lr,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(int(self.max_epochs), 1),
                    eta_min=self.min_lr,
                )
            interval = "epoch"
        else:
            warmup_steps = int(self.warmup_steps or 0)
            total_steps = self._estimate_total_steps()
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
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_steps, 1),
                    eta_min=self.min_lr,
                )
            interval = "step"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }
