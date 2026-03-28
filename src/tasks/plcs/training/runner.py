"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return PLCSDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return PLCSLightningModule(config)

    def apply_runtime_settings(self, config: Any) -> None:
        """Apply PLCS runtime settings from required training config."""
        torch.set_float32_matmul_precision(str(config.training.matmul_precision))

        allow_tf32 = bool(config.training.allow_tf32)
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            deterministic = bool(config.training.trainer.deterministic)
            torch.backends.cudnn.benchmark = bool(config.training.trainer.benchmark and not deterministic)
