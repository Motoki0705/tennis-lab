"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def build_datamodule(self, config: DictConfig) -> pl.LightningDataModule:
        return PLCSDataModule(config)

    def build_lightning_module(
        self,
        config: DictConfig,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return PLCSLightningModule(config)

    def apply_runtime_settings(self, config: DictConfig) -> None:
        """Apply PLCS runtime settings from required training config.

        Expected config structure:
            - config.training.matmul_precision: str
            - config.training.allow_tf32: bool
            - config.training.trainer.deterministic: bool
            - config.training.trainer.benchmark: bool

        This method is called by BaseTrainingRunner.run() before datamodule/model
        creation, so missing keys must fail fast instead of using runtime fallbacks.
        cudnn benchmark is intentionally forced to False when deterministic=True
        because deterministic mode and benchmark autotuning conflict in practice.
        """
        torch.set_float32_matmul_precision(str(config.training.matmul_precision))
        deterministic = bool(config.training.trainer.deterministic)
        benchmark = bool(config.training.trainer.benchmark and not deterministic)

        allow_tf32 = bool(config.training.allow_tf32)
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = benchmark
