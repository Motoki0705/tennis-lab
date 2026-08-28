from __future__ import annotations

import pytest
import torch

from src.tasks.base.generate_dataset import (
    CourtKeypointContractMismatchError,
    MissingCourtKeypointMetadataError,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _module_without_runtime() -> BLCSTrackingLightningModule:
    return object.__new__(BLCSTrackingLightningModule)


def _module_with_runtime(selector: str) -> BLCSTrackingLightningModule:
    module = _module_without_runtime()
    object.__setattr__(
        module,
        "court_keypoint_contract",
        resolve_court_keypoint_contract(selector),
    )
    return module


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


def test_tracking_checkpoint_round_trips_exact_court_keypoint_contract() -> None:
    module = _module_with_runtime("camera_view_v2")
    checkpoint: dict[str, object] = {"state_dict": {}}

    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)

    with pytest.raises(
        CourtKeypointContractMismatchError,
        match="does not exactly match",
    ):
        _module_with_runtime("physical_v1").on_load_checkpoint(checkpoint)


def test_v2_tracking_checkpoint_rejects_missing_court_keypoint_metadata() -> None:
    checkpoint: dict[str, object] = {
        "state_dict": {},
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
    }

    with pytest.raises(MissingCourtKeypointMetadataError, match="metadata is absent"):
        _module_with_runtime("camera_view_v2").on_load_checkpoint(checkpoint)
