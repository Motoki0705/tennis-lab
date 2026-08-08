"""External training composition for BLCS model, adapter, data, and Lightning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytorch_lightning as pl

from src.tasks.blcs.data.datamodule import BLCSDataModule
from src.tasks.blcs.model_io.adapters import (
    TrackQueryModelIOAdapter,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.factory import (
    TrackQueryBoundModelIO,
    TrajectoryBoundModelIO,
    compose_blcs_model_io,
)
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)


@dataclass(frozen=True, slots=True)
class BLCSTrainingComposition:
    """Once-selected runtime objects consumed by the model-agnostic runner."""

    datamodule: pl.LightningDataModule
    lightning_module: pl.LightningModule


def compose_blcs_training(
    config: Any,
) -> BLCSTrainingComposition:
    """Select the complete standard or tracking runtime before any loop starts."""
    binding = compose_blcs_model_io(config)
    backend = str(config.data.backend)
    adapter = binding.adapter
    if isinstance(adapter, TrackQueryModelIOAdapter):
        tracking_binding = cast("TrackQueryBoundModelIO", binding)
        from src.tasks.blcs.data.tracking_datamodule import BLCSTrackingDataModule

        if backend == "default":
            datamodule: pl.LightningDataModule = BLCSTrackingDataModule(config)
        else:
            raise ValueError(
                f"Unsupported tracking data.backend={backend!r}; expected 'default'."
            )
        return BLCSTrainingComposition(
            datamodule=datamodule,
            lightning_module=BLCSTrackingLightningModule(
                config,
                model_io=tracking_binding,
            ),
        )
    if not isinstance(adapter, TrajectoryModelIOAdapter):
        raise TypeError("BLCS composition received an unsupported I/O adapter.")
    trajectory_binding = cast("TrajectoryBoundModelIO", binding)
    collate_fn = adapter.collate_samples
    if backend == "default":
        datamodule = BLCSDataModule(config, collate_fn=collate_fn)
    else:
        raise ValueError(
            f"Unsupported trajectory data.backend={backend!r}; expected 'default'."
        )
    return BLCSTrainingComposition(
        datamodule=datamodule,
        lightning_module=BLCSLightningModule(config, model_io=trajectory_binding),
    )


__all__ = [
    "BLCSTrainingComposition",
    "compose_blcs_training",
]
