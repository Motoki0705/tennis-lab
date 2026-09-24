"""External composition factory for PLCS training lifecycle variants."""

from __future__ import annotations

import json
from typing import Any

import pytorch_lightning as pl

from src.tasks.plcs.configuration import PLCSTrainingConfig, validate_residual_config


def build_plcs_datamodule(config: Any) -> pl.LightningDataModule:
    """Select the validated data lifecycle outside the training runner."""
    if str(config.model.name) in {"plcs_player_reid", "plcs_court_side"}:
        from src.tasks.plcs.data.association_datamodule import PLCSAssociationDataModule

        return PLCSAssociationDataModule(config)

    if config.model.name == "plcs_triangulation_residual":
        from src.tasks.plcs.data.residual_datamodule import ResidualDataModule

        data = ResidualDataModule(validate_residual_config(config))
        data.setup()
        output_dir = data.config.runtime.run.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "split_audit.json").write_text(
            json.dumps(data.split_audit, indent=2, default=str)
        )
        return data
    runtime = PLCSTrainingConfig.from_config(config)
    backend = runtime.data.backend
    if runtime.model.name in {
        "plcs_track_query",
        "plcs_track_query_reference",
    }:
        from src.tasks.plcs.data.tracking_datamodule import (
            ChunkedPLCSTrackingDataModule,
            PLCSTrackingDataModule,
        )

        factories: dict[str, type[pl.LightningDataModule]] = {
            "default": PLCSTrackingDataModule,
            "chunked": ChunkedPLCSTrackingDataModule,
        }
    else:
        from src.tasks.plcs.data.chunked_datamodule import ChunkedPLCSDataModule
        from src.tasks.plcs.data.datamodule import PLCSDataModule

        factories = {
            "default": PLCSDataModule,
            "chunked": ChunkedPLCSDataModule,
        }
    try:
        factory = factories[backend]
    except KeyError as error:
        raise ValueError(
            f"Unsupported PLCS data.backend={backend!r}; expected one of "
            f"{sorted(factories)}."
        ) from error
    return factory(config)


def build_plcs_lightning_module(
    config: Any, *, steps_per_epoch: int | None = None
) -> pl.LightningModule:
    """Select the validated Lightning lifecycle outside the training runner."""
    if str(config.model.name) in {"plcs_player_reid", "plcs_court_side"}:
        from src.tasks.plcs.training.association_lightning_module import (
            PLCSAssociationLightningModule,
        )

        return PLCSAssociationLightningModule(config)

    if config.model.name == "plcs_triangulation_residual":
        from src.tasks.plcs.training.residual_lightning_module import (
            ResidualLightningModule,
        )

        return ResidualLightningModule(config, steps_per_epoch)
    runtime = PLCSTrainingConfig.from_config(config)
    if runtime.model.name in {
        "plcs_track_query",
        "plcs_track_query_reference",
    }:
        from src.tasks.plcs.training.tracking_lightning_module import (
            PLCSTrackingLightningModule,
        )

        return PLCSTrackingLightningModule(config)
    from src.tasks.plcs.training.lightning_module import PLCSLightningModule

    return PLCSLightningModule(config)


__all__ = ["build_plcs_datamodule", "build_plcs_lightning_module"]
