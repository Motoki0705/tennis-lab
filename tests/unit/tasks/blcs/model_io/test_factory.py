"""BLCS ablation model, adapter, and training composition tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.model_io import (
    TrackQueryAblationModelIOAdapter as PublicTrackQueryAblationModelIOAdapter,
)
from src.tasks.blcs.model_io.adapters import TrackQueryAblationModelIOAdapter
from src.tasks.blcs.model_io.factory import (
    TrackQueryBoundModelIO,
    compose_blcs_track_query_model_io,
)
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.blcs.models.blcs_track_query_ablation_model import (
    BLCSTrackQueryAblationModel,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()
_SMALL = (
    "model.hidden_dim=16",
    "model.num_heads=4",
    "model.ffn_dim=32",
    "model.rope_dim=4",
    "model.num_stages=4",
    "model.mhc.coefficient_dim=8",
    "model.mhc.sinkhorn_iters=5",
    "model.cswa.compression_ratio=2",
    "model.cswa.window_radius=1",
)


def _config(condition: str) -> Any:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_tracking",
            overrides=[f"model=track_query_ablation_{condition}", *_SMALL],
        )


def test_ablation_adapter_is_exported_from_canonical_model_io_api() -> None:
    assert PublicTrackQueryAblationModelIOAdapter is TrackQueryAblationModelIOAdapter


@pytest.mark.parametrize("condition", ["a", "b", "c", "d"])
def test_factory_binds_every_ablation_config_to_exact_model_and_adapter(
    condition: str,
) -> None:
    binding = compose_blcs_track_query_model_io(_config(condition))

    assert type(binding.model) is BLCSTrackQueryAblationModel
    assert type(binding.adapter) is TrackQueryAblationModelIOAdapter
    assert binding.adapter.model_type is BLCSTrackQueryAblationModel
    assert binding.adapter.num_queries == 4


def test_ablation_uses_tracking_training_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("d")

    class _DataModule:
        def __init__(self, received: object) -> None:
            self.received = received

    class _LightningModule:
        def __init__(
            self,
            received: object,
            *,
            model_io: TrackQueryBoundModelIO,
        ) -> None:
            self.received = received
            self.model_io = model_io

    monkeypatch.setattr(
        "src.tasks.blcs.data.tracking_datamodule.BLCSTrackingDataModule",
        _DataModule,
    )
    monkeypatch.setattr(
        "src.tasks.blcs.model_io.training.BLCSTrackingLightningModule",
        _LightningModule,
    )

    composition = compose_blcs_training(config, generator_config=None)

    assert type(composition.datamodule) is _DataModule
    assert type(composition.lightning_module) is _LightningModule
    assert composition.datamodule.received is config
    assert composition.lightning_module.received is config
    assert isinstance(composition.lightning_module, _LightningModule)
    assert (
        type(composition.lightning_module.model_io.model)
        is BLCSTrackQueryAblationModel
    )
    assert (
        type(composition.lightning_module.model_io.adapter)
        is TrackQueryAblationModelIOAdapter
    )
