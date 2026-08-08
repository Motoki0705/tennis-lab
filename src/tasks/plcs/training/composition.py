"""External composition factory for PLCS training lifecycle variants."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.plcs.configuration import PLCSTrainingConfig


def build_plcs_datamodule(config: Any) -> pl.LightningDataModule:
    """Select the validated data lifecycle outside the training runner."""
    runtime = PLCSTrainingConfig.from_config(config)
    if runtime.data.backend != "default":
        raise ValueError("PLCS training requires the canonical fixed-path backend.")
    if runtime.model.name == "plcs_track_query":
        from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule

        factory: type[pl.LightningDataModule] = PLCSTrackingDataModule
    else:
        from src.tasks.plcs.data.datamodule import PLCSDataModule

        factory = PLCSDataModule
    return factory(config)


def build_plcs_lightning_module(config: Any) -> pl.LightningModule:
    """Select the validated Lightning lifecycle outside the training runner."""
    runtime = PLCSTrainingConfig.from_config(config)
    if runtime.model.name == "plcs_track_query":
        from src.tasks.plcs.training.tracking_lightning_module import (
            PLCSTrackingLightningModule,
        )

        return PLCSTrackingLightningModule(config)
    from src.tasks.plcs.training.lightning_module import PLCSLightningModule

    return PLCSLightningModule(config)


__all__ = ["build_plcs_datamodule", "build_plcs_lightning_module"]
