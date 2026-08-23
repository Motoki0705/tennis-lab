"""Immutable, versioned normalization for physical court coordinates.

The physical court geometry remains defined in :mod:`src.utils.schema.court`.
This module is the sole mapping from a normalization version to its XYZ scale
and the sole implementation of position/velocity conversion. Callers select a
contract explicitly and pass it through their runtime; there is no mutable
process-wide active version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, overload

import numpy as np
import torch
from torch import Tensor

CourtCoordinateNormalizationVersion: TypeAlias = Literal["v1", "v2"]
CourtCoordinateArray: TypeAlias = np.ndarray | Tensor

__all__ = [
    "CourtCoordinateArray",
    "CourtCoordinateNormalization",
    "CourtCoordinateNormalizationError",
    "CourtCoordinateNormalizationVersion",
    "CourtCoordinateShapeError",
    "UnknownCourtCoordinateNormalizationVersionError",
    "denormalize_court_position",
    "denormalize_court_velocity",
    "normalize_court_position",
    "normalize_court_velocity",
    "resolve_court_coordinate_normalization",
]


class CourtCoordinateNormalizationError(ValueError):
    """Base error for an invalid court-coordinate normalization contract."""


class UnknownCourtCoordinateNormalizationVersionError(
    CourtCoordinateNormalizationError
):
    """Raised when a requested normalization version is not in the schema."""


class CourtCoordinateShapeError(CourtCoordinateNormalizationError):
    """Raised when a position/velocity value does not have trailing XYZ axes."""


@dataclass(frozen=True, slots=True)
class CourtCoordinateNormalization:
    """One immutable normalization contract.

    ``scale_xyz`` is measured in metres. It applies to physical positions in
    metres and physical velocities in metres per second; normalized values are
    dimensionless and dimensionless per second, respectively.
    """

    version: CourtCoordinateNormalizationVersion
    scale_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        canonical_version, canonical_scale = _normalization_definition(self.version)
        if type(self.scale_xyz) is not tuple or len(self.scale_xyz) != 3:
            raise CourtCoordinateNormalizationError(
                "court-coordinate normalization scale_xyz must be an immutable "
                f"three-float tuple; got {self.scale_xyz!r}."
            )
        if any(type(value) is not float for value in self.scale_xyz):
            raise CourtCoordinateNormalizationError(
                "court-coordinate normalization scale_xyz must contain exactly "
                f"three floats; got {self.scale_xyz!r}."
            )
        if any(not np.isfinite(value) or value <= 0.0 for value in self.scale_xyz):
            raise CourtCoordinateNormalizationError(
                "court-coordinate normalization scale_xyz must contain finite, "
                f"positive values; got {self.scale_xyz!r}."
            )
        if self.version != canonical_version or self.scale_xyz != canonical_scale:
            raise CourtCoordinateNormalizationError(
                f"Court-coordinate normalization {self.version!r} must use the "
                f"canonical resolver scale {canonical_scale!r}; got "
                f"{self.scale_xyz!r}."
            )

    @overload
    def normalize_position(self, value: np.ndarray) -> np.ndarray: ...

    @overload
    def normalize_position(self, value: Tensor) -> Tensor: ...

    def normalize_position(self, value: CourtCoordinateArray) -> CourtCoordinateArray:
        """Convert physical position ``(..., 3)`` in metres to normalized space."""
        return _apply_scale(value, self.scale_xyz, divide=True, quantity="position")

    @overload
    def denormalize_position(self, value: np.ndarray) -> np.ndarray: ...

    @overload
    def denormalize_position(self, value: Tensor) -> Tensor: ...

    def denormalize_position(
        self, value: CourtCoordinateArray
    ) -> CourtCoordinateArray:
        """Convert normalized position ``(..., 3)`` to physical metres."""
        return _apply_scale(value, self.scale_xyz, divide=False, quantity="position")

    @overload
    def normalize_velocity(self, value: np.ndarray) -> np.ndarray: ...

    @overload
    def normalize_velocity(self, value: Tensor) -> Tensor: ...

    def normalize_velocity(self, value: CourtCoordinateArray) -> CourtCoordinateArray:
        """Convert physical velocity ``(..., 3)`` in m/s to normalized units/s."""
        return _apply_scale(value, self.scale_xyz, divide=True, quantity="velocity")

    @overload
    def denormalize_velocity(self, value: np.ndarray) -> np.ndarray: ...

    @overload
    def denormalize_velocity(self, value: Tensor) -> Tensor: ...

    def denormalize_velocity(
        self, value: CourtCoordinateArray
    ) -> CourtCoordinateArray:
        """Convert normalized velocity ``(..., 3)`` to physical m/s."""
        return _apply_scale(value, self.scale_xyz, divide=False, quantity="velocity")


def _validate_shape(value: CourtCoordinateArray, *, quantity: str) -> None:
    if value.ndim < 1 or value.shape[-1] != 3:
        raise CourtCoordinateShapeError(
            f"Court {quantity} must have shape (..., 3); got {tuple(value.shape)!r}."
        )


@overload
def _apply_scale(
    value: np.ndarray,
    scale_xyz: tuple[float, float, float],
    *,
    divide: bool,
    quantity: str,
) -> np.ndarray: ...


@overload
def _apply_scale(
    value: Tensor,
    scale_xyz: tuple[float, float, float],
    *,
    divide: bool,
    quantity: str,
) -> Tensor: ...


def _apply_scale(
    value: CourtCoordinateArray,
    scale_xyz: tuple[float, float, float],
    *,
    divide: bool,
    quantity: str,
) -> CourtCoordinateArray:
    if not isinstance(value, (np.ndarray, Tensor)):
        raise TypeError(
            f"Court {quantity} must be a numpy.ndarray or torch.Tensor; "
            f"got {type(value).__name__}."
        )
    _validate_shape(value, quantity=quantity)

    if isinstance(value, Tensor):
        torch_dtype = (
            value.dtype if value.is_floating_point() else torch.get_default_dtype()
        )
        torch_scale = torch.tensor(
            scale_xyz,
            dtype=torch_dtype,
            device=value.device,
        )
        return value / torch_scale if divide else value * torch_scale

    numpy_dtype = (
        value.dtype if np.issubdtype(value.dtype, np.floating) else np.float64
    )
    numpy_scale = np.asarray(scale_xyz, dtype=numpy_dtype)
    numpy_result: np.ndarray = (
        value / numpy_scale if divide else value * numpy_scale
    )
    return numpy_result


def _normalization_definition(
    version: str,
) -> tuple[
    CourtCoordinateNormalizationVersion,
    tuple[float, float, float],
]:
    """Return the sole version-to-scale mapping used by construction/resolution."""
    # Import lazily so ``court.py`` can expose legacy aliases derived from the
    # resolver while remaining the physical-dimension authority.
    from src.utils.schema.court import (
        HALF_DOUBLES_WIDTH,
        HALF_LENGTH,
        NET_HEIGHT_POST,
    )

    if version == "v1":
        return (
            "v1",
            (
                float(HALF_DOUBLES_WIDTH),
                float(HALF_LENGTH),
                float(NET_HEIGHT_POST),
            ),
        )
    if version == "v2":
        common_scale = float(HALF_LENGTH)
        return (
            "v2",
            (common_scale, common_scale, common_scale),
        )
    raise UnknownCourtCoordinateNormalizationVersionError(
        f"Unknown court-coordinate normalization version: {version!r}. "
        "Supported versions are 'v1' and 'v2'."
    )


def resolve_court_coordinate_normalization(
    version: str,
) -> CourtCoordinateNormalization:
    """Resolve one supported version without inferring from data or global state."""
    canonical_version, scale_xyz = _normalization_definition(version)
    return CourtCoordinateNormalization(
        version=canonical_version,
        scale_xyz=scale_xyz,
    )


@overload
def normalize_court_position(
    value: np.ndarray, contract: CourtCoordinateNormalization
) -> np.ndarray: ...


@overload
def normalize_court_position(
    value: Tensor, contract: CourtCoordinateNormalization
) -> Tensor: ...


def normalize_court_position(
    value: CourtCoordinateArray,
    contract: CourtCoordinateNormalization,
) -> CourtCoordinateArray:
    """Normalize position using an explicitly supplied contract."""
    return contract.normalize_position(value)


@overload
def denormalize_court_position(
    value: np.ndarray, contract: CourtCoordinateNormalization
) -> np.ndarray: ...


@overload
def denormalize_court_position(
    value: Tensor, contract: CourtCoordinateNormalization
) -> Tensor: ...


def denormalize_court_position(
    value: CourtCoordinateArray,
    contract: CourtCoordinateNormalization,
) -> CourtCoordinateArray:
    """Denormalize position using an explicitly supplied contract."""
    return contract.denormalize_position(value)


@overload
def normalize_court_velocity(
    value: np.ndarray, contract: CourtCoordinateNormalization
) -> np.ndarray: ...


@overload
def normalize_court_velocity(
    value: Tensor, contract: CourtCoordinateNormalization
) -> Tensor: ...


def normalize_court_velocity(
    value: CourtCoordinateArray,
    contract: CourtCoordinateNormalization,
) -> CourtCoordinateArray:
    """Normalize velocity using an explicitly supplied contract."""
    return contract.normalize_velocity(value)


@overload
def denormalize_court_velocity(
    value: np.ndarray, contract: CourtCoordinateNormalization
) -> np.ndarray: ...


@overload
def denormalize_court_velocity(
    value: Tensor, contract: CourtCoordinateNormalization
) -> Tensor: ...


def denormalize_court_velocity(
    value: CourtCoordinateArray,
    contract: CourtCoordinateNormalization,
) -> CourtCoordinateArray:
    """Denormalize velocity using an explicitly supplied contract."""
    return contract.denormalize_velocity(value)
