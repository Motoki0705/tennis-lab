"""Trajectory completion training runner using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.trajectory_completion.data.datamodule import TrajectoryCompletionDataModule
from src.trajectory_completion.training.lightning_module import (
    TrajectoryCompletionLightningModule,
)


class TrajectoryCompletionTrainingRunner(BaseTrainingRunner):
    """Training runner for trajectory completion model."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build trajectory completion data module."""
        return TrajectoryCompletionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build trajectory completion lightning module."""
        return TrajectoryCompletionLightningModule(config)

    def checkpoint_prefix(self, config: Any) -> str:
        """Return checkpoint filename prefix."""
        return "trajectory-completion"

    def checkpoint_monitor(self, config: Any) -> str:
        """Return metric to monitor for checkpointing."""
        return "val/loss"

    def early_stopping_monitor(self, config: Any) -> str:
        """Return metric to monitor for early stopping."""
        return "val/loss"

    def early_stopping_patience(self, config: Any) -> int:
        """Return early stopping patience."""
        return 5

    def early_stopping_min_delta(self, config: Any) -> float | None:
        """Return early stopping min_delta."""
        return 1.0e-3

    def trainer_kwargs(
        self, config: Any, accelerator: str, devices: int
    ) -> dict[str, Any]:
        """Return additional trainer kwargs."""
        kwargs: dict[str, Any] = {
            "log_every_n_steps": 50,
        }
        if accelerator == "gpu":
            kwargs["precision"] = "16-mixed"
        return kwargs

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Log batch shapes after dry run batch loading."""
        pass
