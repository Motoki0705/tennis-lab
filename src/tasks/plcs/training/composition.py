"""Compose axial PLCS training with fixed or generated single-object data."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.chunked_datamodule import ChunkedPLCSDataModule
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule


def build_plcs_datamodule(config: Any) -> pl.LightningDataModule:
    runtime = PLCSTrainingConfig.from_config(config)
    factories = {"default": PLCSDataModule, "chunked": ChunkedPLCSDataModule}
    try:
        factory = factories[runtime.data.backend]
    except KeyError as error:
        raise ValueError(
            f"Unsupported PLCS data.backend={runtime.data.backend!r}."
        ) from error
    return factory(config)


def build_plcs_lightning_module(
    config: Any, *, steps_per_epoch: int | None = None
) -> pl.LightningModule:
    PLCSTrainingConfig.from_config(config)
    return PLCSLightningModule(config)


__all__ = ["build_plcs_datamodule", "build_plcs_lightning_module"]
