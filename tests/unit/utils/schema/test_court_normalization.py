"""Contract tests for versioned physical court-coordinate normalization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    CourtCoordinateNormalizationError,
    CourtCoordinateShapeError,
    UnknownCourtCoordinateNormalizationVersionError,
    resolve_court_coordinate_normalization,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("v1", (5.485, 11.885, 1.07)),
        ("v2", (11.885, 11.885, 11.885)),
    ],
)
def test_resolver_is_the_single_version_to_scale_mapping(
    version: str,
    expected: tuple[float, float, float],
) -> None:
    contract = resolve_court_coordinate_normalization(version)

    assert contract.version == version
    assert contract.scale_xyz == expected


def test_v1_mapping_remains_the_physical_geometry_golden() -> None:
    assert resolve_court_coordinate_normalization("v1").scale_xyz == (
        HALF_DOUBLES_WIDTH,
        HALF_LENGTH,
        NET_HEIGHT_POST,
    )


def test_unknown_version_fails_without_inference() -> None:
    with pytest.raises(
        UnknownCourtCoordinateNormalizationVersionError,
        match="Supported versions are 'v1' and 'v2'",
    ):
        resolve_court_coordinate_normalization("latest")


def test_contract_cannot_be_mutated_or_constructed_with_noncanonical_scale() -> None:
    contract = resolve_court_coordinate_normalization("v2")
    with pytest.raises(FrozenInstanceError):
        contract.version = "v1"  # type: ignore[misc]

    with pytest.raises(CourtCoordinateNormalizationError, match="canonical resolver"):
        CourtCoordinateNormalization("v2", (1.0, 1.0, 1.0))


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 5, 3)])
def test_numpy_round_trip_is_shape_generic_and_preserves_float_dtype(
    version: str,
    shape: tuple[int, ...],
) -> None:
    values = np.linspace(-25.0, 25.0, int(np.prod(shape)), dtype=np.float32).reshape(
        shape
    )
    contract = resolve_court_coordinate_normalization(version)

    normalized = contract.normalize_position(values)
    restored = contract.denormalize_position(normalized)

    assert normalized.shape == shape
    assert normalized.dtype == np.float32
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, values, atol=1.0e-5, rtol=0.0)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 5, 3)])
def test_torch_position_and_velocity_round_trip_preserve_dtype_and_device(
    version: str,
    shape: tuple[int, ...],
) -> None:
    values = torch.linspace(-25.0, 25.0, int(np.prod(shape)), dtype=torch.float64).reshape(
        shape
    )
    contract = resolve_court_coordinate_normalization(version)

    position = contract.denormalize_position(contract.normalize_position(values))
    velocity = contract.denormalize_velocity(contract.normalize_velocity(values))

    assert position.dtype == values.dtype
    assert position.device == values.device
    torch.testing.assert_close(position, values, atol=1.0e-10, rtol=0.0)
    torch.testing.assert_close(velocity, values, atol=1.0e-10, rtol=0.0)


@pytest.mark.parametrize("value", [np.zeros((2, 2)), torch.zeros(2, 2), np.array(1.0)])
def test_conversion_rejects_values_without_trailing_xyz(value: object) -> None:
    with pytest.raises(CourtCoordinateShapeError, match=r"shape \(\.\.\., 3\)"):
        resolve_court_coordinate_normalization("v2").normalize_position(value)  # type: ignore[arg-type]


def test_integer_inputs_are_promoted_instead_of_truncated() -> None:
    contract = resolve_court_coordinate_normalization("v2")
    numpy_result = contract.normalize_position(np.array([1, 2, 3], dtype=np.int64))
    torch_result = contract.normalize_position(torch.tensor([1, 2, 3]))

    assert numpy_result.dtype == np.float64
    assert torch_result.is_floating_point()
    assert bool((numpy_result != 0.0).all())
    assert bool((torch_result != 0.0).all())


def test_v2_preserves_physical_court_aspect_ratio_in_normalized_space() -> None:
    contract = resolve_court_coordinate_normalization("v2")
    landmarks_m = np.array(
        [
            [HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST],
            [-HALF_DOUBLES_WIDTH, -HALF_LENGTH, 0.0],
        ],
        dtype=np.float64,
    )

    normalized = contract.normalize_position(landmarks_m)

    np.testing.assert_allclose(
        normalized[0],
        [HALF_DOUBLES_WIDTH / HALF_LENGTH, 1.0, NET_HEIGHT_POST / HALF_LENGTH],
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        normalized[1],
        [-HALF_DOUBLES_WIDTH / HALF_LENGTH, -1.0, 0.0],
        atol=1.0e-12,
        rtol=0.0,
    )
