"""Single numeric contract for normalized court coordinates.

Normalized court positions and velocities divide every physical XYZ component
by the centre-to-baseline distance.  Physical court geometry remains owned by
``src.utils.schema.court`` and is not changed by this representation contract.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch
from torch import Tensor

from src.utils.schema.court import COURT_COORD_SCALE_XYZ

COURT_COORDINATE_NORMALIZATION_IDENTITY = "isotropic_half_length"
COURT_COORDINATE_NORMALIZATION_KEY = "court_coordinate_normalization"
_POSITION_UNIT = "m / scale_xyz_m"
_VELOCITY_UNIT = "m/s / scale_xyz_m"
_FIELDS = frozenset({"identity", "scale_xyz_m", "position_unit", "velocity_unit"})

ArrayT = TypeVar("ArrayT", np.ndarray, Tensor)


class CourtCoordinateContractError(ValueError):
    """Raised when a serialized normalized-coordinate artifact is incompatible."""


def _validate_coordinates(value: np.ndarray | Tensor, *, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {tuple(value.shape)}.")
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype, got {value.dtype}.")
    elif not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}.")


def _normalize(value: ArrayT, *, name: str) -> ArrayT:
    _validate_coordinates(value, name=name)
    if isinstance(value, Tensor):
        tensor_result: Tensor = value / value.new_tensor(COURT_COORD_SCALE_XYZ)
        return cast(ArrayT, tensor_result)
    array_result: np.ndarray = np.divide(
        value,
        np.asarray(COURT_COORD_SCALE_XYZ, dtype=value.dtype),
    )
    return array_result


def _denormalize(value: ArrayT, *, name: str) -> ArrayT:
    _validate_coordinates(value, name=name)
    if isinstance(value, Tensor):
        tensor_result: Tensor = value * value.new_tensor(COURT_COORD_SCALE_XYZ)
        return cast(ArrayT, tensor_result)
    array_result: np.ndarray = np.multiply(
        value,
        np.asarray(COURT_COORD_SCALE_XYZ, dtype=value.dtype),
    )
    return array_result


def normalize_court_position(position_m: ArrayT) -> ArrayT:
    """Convert metre-valued ``(..., 3)`` positions to the fixed contract."""
    normalized: ArrayT = _normalize(position_m, name="position_m")
    return normalized


def denormalize_court_position(position_norm: ArrayT) -> ArrayT:
    """Convert fixed-contract ``(..., 3)`` positions to metres."""
    denormalized: ArrayT = _denormalize(position_norm, name="position_norm")
    return denormalized


def normalize_court_velocity(velocity_mps: ArrayT) -> ArrayT:
    """Convert metre-per-second ``(..., 3)`` velocities to the fixed contract."""
    normalized: ArrayT = _normalize(velocity_mps, name="velocity_mps")
    return normalized


def denormalize_court_velocity(velocity_norm: ArrayT) -> ArrayT:
    """Convert fixed-contract ``(..., 3)`` velocities to metres per second."""
    denormalized: ArrayT = _denormalize(velocity_norm, name="velocity_norm")
    return denormalized


def court_coordinate_normalization_metadata() -> dict[str, object]:
    """Return a fresh JSON-compatible mapping for the sole current contract."""
    return {
        "identity": COURT_COORDINATE_NORMALIZATION_IDENTITY,
        "scale_xyz_m": list(COURT_COORD_SCALE_XYZ),
        "position_unit": _POSITION_UNIT,
        "velocity_unit": _VELOCITY_UNIT,
    }


def validate_court_coordinate_normalization(
    container: Mapping[str, Any],
    *,
    artifact: str,
) -> None:
    """Require the exact fixed contract under the canonical serialized key."""
    if COURT_COORDINATE_NORMALIZATION_KEY not in container:
        raise CourtCoordinateContractError(
            f"{artifact} is incompatible: missing "
            f"{COURT_COORDINATE_NORMALIZATION_KEY!r}; regenerate or retrain it."
        )
    raw = container[COURT_COORDINATE_NORMALIZATION_KEY]
    if not isinstance(raw, Mapping):
        raise CourtCoordinateContractError(
            f"{artifact} is incompatible: {COURT_COORDINATE_NORMALIZATION_KEY!r} "
            "must be a mapping; regenerate or retrain it."
        )
    if set(raw) != _FIELDS:
        raise CourtCoordinateContractError(
            f"{artifact} is incompatible: {COURT_COORDINATE_NORMALIZATION_KEY!r} "
            f"must contain exactly {sorted(_FIELDS)}; regenerate or retrain it."
        )
    scale = raw["scale_xyz_m"]
    if not isinstance(scale, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in scale
    ):
        raise CourtCoordinateContractError(
            f"{artifact} is incompatible: scale_xyz_m must be a numeric JSON list."
        )
    expected = court_coordinate_normalization_metadata()
    actual = {
        "identity": raw["identity"],
        "scale_xyz_m": [float(value) for value in scale],
        "position_unit": raw["position_unit"],
        "velocity_unit": raw["velocity_unit"],
    }
    if actual != expected:
        raise CourtCoordinateContractError(
            f"{artifact} uses an unknown or mismatched court coordinate "
            "normalization contract; regenerate or retrain it."
        )


def add_court_coordinate_normalization(
    container: MutableMapping[str, Any],
    *,
    artifact: str,
) -> None:
    """Attach the fixed mapping, rejecting a pre-existing conflicting value."""
    if COURT_COORDINATE_NORMALIZATION_KEY in container:
        validate_court_coordinate_normalization(container, artifact=artifact)
        return
    container[COURT_COORDINATE_NORMALIZATION_KEY] = (
        court_coordinate_normalization_metadata()
    )


def load_and_validate_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load a raw checkpoint on CPU and validate its contract before composition."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise CourtCoordinateContractError(
            f"Checkpoint {path} is incompatible: root must be a mapping."
        )
    validate_court_coordinate_normalization(checkpoint, artifact=f"Checkpoint {path}")
    return checkpoint


__all__ = [
    "COURT_COORDINATE_NORMALIZATION_IDENTITY",
    "COURT_COORDINATE_NORMALIZATION_KEY",
    "CourtCoordinateContractError",
    "add_court_coordinate_normalization",
    "court_coordinate_normalization_metadata",
    "denormalize_court_position",
    "denormalize_court_velocity",
    "load_and_validate_checkpoint",
    "normalize_court_position",
    "normalize_court_velocity",
    "validate_court_coordinate_normalization",
]
