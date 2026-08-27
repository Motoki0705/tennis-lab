"""Unit tests for the single numeric and serialized normalized-court contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.schema.court import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_XYZ,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)
from src.utils.schema.court_normalization import (
    COURT_COORDINATE_NORMALIZATION_IDENTITY,
    COURT_COORDINATE_NORMALIZATION_KEY,
    CourtCoordinateContractError,
    add_court_coordinate_normalization,
    court_coordinate_normalization_metadata,
    denormalize_court_position,
    denormalize_court_velocity,
    load_and_validate_checkpoint,
    normalize_court_position,
    normalize_court_velocity,
    validate_court_coordinate_normalization,
)


def _container() -> dict[str, object]:
    return {
        COURT_COORDINATE_NORMALIZATION_KEY: (court_coordinate_normalization_metadata())
    }


def test_fixed_scale_is_derived_from_half_length_only() -> None:
    assert COURT_COORD_SCALE_XYZ == (HALF_LENGTH, HALF_LENGTH, HALF_LENGTH)
    assert COURT_COORD_SCALE_X == HALF_LENGTH
    assert COURT_COORD_SCALE_Y == HALF_LENGTH
    assert COURT_COORD_SCALE_Z == HALF_LENGTH
    assert COURT_COORDINATE_NORMALIZATION_IDENTITY == "isotropic_half_length"


@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 4, 3)])
@pytest.mark.parametrize("quantity", ["position", "velocity"])
def test_numpy_round_trip_preserves_shape_dtype_and_values(
    shape: tuple[int, ...], quantity: str
) -> None:
    values = np.linspace(-12.0, 12.0, int(np.prod(shape)), dtype=np.float32).reshape(
        shape
    )
    normalize = (
        normalize_court_position if quantity == "position" else normalize_court_velocity
    )
    denormalize = (
        denormalize_court_position
        if quantity == "position"
        else denormalize_court_velocity
    )

    normalized = normalize(values)
    recovered = denormalize(normalized)

    assert normalized.shape == shape
    assert normalized.dtype == values.dtype
    assert recovered.dtype == values.dtype
    np.testing.assert_allclose(recovered, values, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("quantity", ["position", "velocity"])
def test_torch_round_trip_preserves_dtype_device_and_values(quantity: str) -> None:
    values = torch.linspace(-12.0, 12.0, 24, dtype=torch.float64).reshape(2, 4, 3)
    normalize = (
        normalize_court_position if quantity == "position" else normalize_court_velocity
    )
    denormalize = (
        denormalize_court_position
        if quantity == "position"
        else denormalize_court_velocity
    )

    normalized = normalize(values)
    recovered = denormalize(normalized)

    assert normalized.dtype == values.dtype
    assert normalized.device == values.device
    torch.testing.assert_close(recovered, values, atol=1e-5, rtol=0.0)


def test_physical_landmarks_use_isotropic_normalization() -> None:
    landmarks_m = np.asarray(
        [
            [HALF_DOUBLES_WIDTH, 0.0, 0.0],
            [-HALF_DOUBLES_WIDTH, 0.0, 0.0],
            [0.0, HALF_LENGTH, 0.0],
            [0.0, -HALF_LENGTH, 0.0],
            [0.0, 0.0, NET_HEIGHT_POST],
        ],
        dtype=np.float64,
    )

    normalized = normalize_court_position(landmarks_m)

    assert normalized[0, 0] == pytest.approx(0.4615061, abs=1e-6)
    assert normalized[1, 0] == pytest.approx(-0.4615061, abs=1e-6)
    assert normalized[2, 1] == 1.0
    assert normalized[3, 1] == -1.0
    assert normalized[4, 2] == pytest.approx(0.0900294, abs=1e-6)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 2), dtype=np.float32),
        torch.zeros(2, 4),
    ],
)
def test_wrong_final_dimension_is_rejected(value: np.ndarray | torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        normalize_court_position(value)  # type: ignore[type-var]


@pytest.mark.parametrize(
    "value",
    [np.zeros((2, 3), dtype=np.int64), torch.zeros(2, 3, dtype=torch.int64)],
)
def test_non_floating_dtype_is_rejected(value: np.ndarray | torch.Tensor) -> None:
    with pytest.raises(TypeError, match="floating dtype"):
        normalize_court_velocity(value)  # type: ignore[type-var]


def test_exact_serialized_mapping() -> None:
    assert court_coordinate_normalization_metadata() == {
        "identity": "isotropic_half_length",
        "scale_xyz_m": [11.885, 11.885, 11.885],
        "position_unit": "m / scale_xyz_m",
        "velocity_unit": "m/s / scale_xyz_m",
    }
    validate_court_coordinate_normalization(_container(), artifact="scene")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "nonmapping",
        "unknown_identity",
        "partial",
        "mismatched_scale",
        "mismatched_position_unit",
        "mismatched_velocity_unit",
        "extra_field",
    ],
)
def test_invalid_scene_contracts_fail_loudly(mutation: str) -> None:
    container = _container()
    raw = deepcopy(container[COURT_COORDINATE_NORMALIZATION_KEY])
    assert isinstance(raw, dict)
    if mutation == "missing":
        container.clear()
    elif mutation == "nonmapping":
        container[COURT_COORDINATE_NORMALIZATION_KEY] = "isotropic_half_length"
    elif mutation == "unknown_identity":
        raw["identity"] = "anisotropic"
        container[COURT_COORDINATE_NORMALIZATION_KEY] = raw
    elif mutation == "partial":
        raw.pop("velocity_unit")
        container[COURT_COORDINATE_NORMALIZATION_KEY] = raw
    elif mutation == "mismatched_scale":
        raw["scale_xyz_m"] = [5.485, 11.885, 1.07]
        container[COURT_COORDINATE_NORMALIZATION_KEY] = raw
    elif mutation == "mismatched_position_unit":
        raw["position_unit"] = "m"
        container[COURT_COORDINATE_NORMALIZATION_KEY] = raw
    elif mutation == "mismatched_velocity_unit":
        raw["velocity_unit"] = "m/s"
        container[COURT_COORDINATE_NORMALIZATION_KEY] = raw
    else:
        raw["version"] = 2
        container[COURT_COORDINATE_NORMALIZATION_KEY] = raw

    with pytest.raises(CourtCoordinateContractError, match="incompatible|unknown"):
        validate_court_coordinate_normalization(container, artifact="scene")


def test_checkpoint_writer_attaches_and_rejects_conflicts() -> None:
    checkpoint: dict[str, object] = {}
    add_court_coordinate_normalization(checkpoint, artifact="checkpoint")
    assert checkpoint == _container()

    conflicting = _container()
    raw = conflicting[COURT_COORDINATE_NORMALIZATION_KEY]
    assert isinstance(raw, dict)
    raw["identity"] = "other"
    with pytest.raises(CourtCoordinateContractError, match="unknown"):
        add_court_coordinate_normalization(conflicting, artifact="checkpoint")


def test_raw_checkpoint_validation_happens_without_inference(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.ckpt"
    torch.save({**_container(), "state_dict": {}}, valid_path)
    assert "state_dict" in load_and_validate_checkpoint(valid_path)

    old_path = tmp_path / "old.ckpt"
    torch.save({"state_dict": {}}, old_path)
    with pytest.raises(CourtCoordinateContractError, match="missing"):
        load_and_validate_checkpoint(old_path)

    malformed_path = tmp_path / "malformed.ckpt"
    torch.save([], malformed_path)
    with pytest.raises(CourtCoordinateContractError, match="root must be a mapping"):
        load_and_validate_checkpoint(malformed_path)


@pytest.mark.parametrize(
    "module_type",
    [
        BLCSLightningModule,
        BLCSTrackingLightningModule,
        PLCSLightningModule,
        PLCSTrackingLightningModule,
    ],
)
def test_task_lightning_save_hooks_write_the_exact_contract(module_type: Any) -> None:
    checkpoint: dict[str, object] = {}
    module_type.on_save_checkpoint(object(), checkpoint)
    assert checkpoint == _container()


@pytest.mark.parametrize(
    "module_type",
    [BLCSLightningModule, PLCSLightningModule, PLCSTrackingLightningModule],
)
def test_task_lightning_load_hooks_reject_old_checkpoints(module_type: Any) -> None:
    with pytest.raises(CourtCoordinateContractError, match="missing"):
        module_type.on_load_checkpoint(object(), {})
