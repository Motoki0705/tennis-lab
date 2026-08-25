from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _module_without_runtime() -> BLCSTrackingLightningModule:
    return object.__new__(BLCSTrackingLightningModule)


def test_checkpoint_rejects_deleted_group_encoder_contract() -> None:
    checkpoint = {
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
        "state_dict": {
            "model.group_encoder.proj.layers.0.weight": torch.randn(3, 2),
        },
    }

    with pytest.raises(RuntimeError, match="deleted model.group_encoder"):
        _module_without_runtime().on_load_checkpoint(checkpoint)


def test_checkpoint_requires_explicit_state_dict_mapping() -> None:
    with pytest.raises(TypeError, match="state_dict mapping"):
        _module_without_runtime().on_load_checkpoint(
            {
                "court_coordinate_normalization": (
                    court_coordinate_normalization_metadata()
                )
            }
        )
