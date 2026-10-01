"""Training runner for ball detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch

from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner


class BallDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for ball detection.

    Overrides datamodule/model construction for ball detection.
    """

    def maybe_load_init_weights(
        self, config: TrainingRuntimeConfig, lightning_module: pl.LightningModule,
    ) -> None:
        """Strictly transfer 2D detector weights without a 3D court contract.

        The checkpoint is an explicitly configured trusted local input.
        Optimizer/scheduler/epoch state is intentionally not restored. Require
        all module tensors, including a discriminator when configured; changing
        the module topology is not an implicit partial transfer.
        """
        path = config.run.init_weights
        if path is None:
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Ball init_weights checkpoint {path} must be a mapping.")
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(
                f"Ball init_weights checkpoint {path} requires a nonempty state_dict mapping."
            )
        lightning_module.load_state_dict(state, strict=True)
        print(f"[ball init_weights] strictly loaded {len(state)} tensors from {path}")

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build the configured ball detection DataModule."""
        return build_ball_detection_datamodule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build BallDetectionLightningModule from config."""
        return BallDetectionLightningModule(config)
