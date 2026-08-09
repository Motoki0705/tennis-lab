"""Canonical semantic scene, camera, alignment, and multi-court contracts.

These contracts deliberately describe geometry and observable validation only.
Artifact hashes, repository revisions, provider internals, and compatibility
schemas are not part of scene validity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

CAMERA_AXES_OPENCV = "opencv:+x_right,+y_down,+z_forward"
COURT_AXES_METRES = "right_handed_metres:+x_right_sideline,+y_far_baseline,+z_up"
PIXEL_COORDINATES = "undistorted_zero_based_pixel_centres"
SCENE_CONTRACT_SCHEMA = "tennis_scene_semantic_v1"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MATRIX_ATOL = 1.0e-6


def _validate_id(value: str, *, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty portable identifier: {value!r}.")


def _finite_tuple(value: Sequence[object], *, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} numeric values.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain only numeric values.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values.")
        result.append(number)
    return tuple(result)


def _strict_mapping(value: object, *, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys do not match the schema; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}."
        )
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


def _validate_rotation(rotation: NDArray[np.float64], *, name: str) -> None:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must be a proper rotation with determinant +1.")


def _validate_rigid(matrix: NDArray[np.float64], *, name: str) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1].")
    _validate_rotation(matrix[:3, :3], name=f"{name} rotation")


def _json_value(value: object, *, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings.")
        return {key: _json_value(item, name=f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item, name=name) for item in value]
    raise TypeError(f"{name} must be JSON-compatible, got {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A finite proper SE(3) transform acting on column-vector points."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = _finite_tuple(self.values, size=16, name="rigid transform")
        _validate_rigid(np.asarray(values, dtype=np.float64).reshape(4, 4), name="rigid transform")
        object.__setattr__(self, "values", values)

    @classmethod
    def identity(cls) -> RigidTransform:
        """Return the identity transform."""
        return cls(tuple(float(value) for value in np.eye(4).ravel()))

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.floating[Any]]) -> RigidTransform:
        """Validate and construct from a ``(4, 4)`` numeric array."""
        array = np.asarray(matrix, dtype=np.float64)
        _validate_rigid(array, name="rigid transform")
        return cls(tuple(float(value) for value in array.ravel()))

    def matrix(self) -> NDArray[np.float64]:
        """Return a new ``(4, 4)`` float64 matrix."""
        return np.asarray(self.values, dtype=np.float64).reshape(4, 4).copy()

    def inverse(self) -> RigidTransform:
        """Return the exact inverse rigid transform."""
        matrix = self.matrix()
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = matrix[:3, :3].T
        inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
        return RigidTransform.from_matrix(inverse)

    def apply(self, points: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Apply the transform to finite ``(..., 3)`` points."""
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != 3 or not np.isfinite(array).all():
            raise ValueError(f"points must be a finite (..., 3) array, got {array.shape}.")
        matrix = self.matrix()
        return array @ matrix[:3, :3].T + matrix[:3, 3]

    def to_list(self) -> list[float]:
        """Serialize the transform without an identity token."""
        return list(self.values)


@dataclass(frozen=True, slots=True)
class SceneCamera:
    """One PINHOLE camera in the standard OpenCV scene convention."""

    camera_id: str
    source_frame_index: int
    width: int
    height: int
    intrinsics: tuple[float, ...]
    camera_to_scene: RigidTransform
    image_path: str

    def __post_init__(self) -> None:
        _validate_id(self.camera_id, name="camera_id")
        _integer(self.source_frame_index, name="source_frame_index")
        _integer(self.width, name="width", minimum=2)
        _integer(self.height, name="height", minimum=2)
        image_path = _string(self.image_path, name="image_path")
        intrinsics = _finite_tuple(self.intrinsics, size=9, name="intrinsics")
        matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")
        if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=_MATRIX_ATOL, rtol=0.0):
            raise ValueError("Camera intrinsics must have bottom row [0, 0, 1].")
        if not 0.0 <= matrix[0, 2] < self.width or not 0.0 <= matrix[1, 2] < self.height:
            raise ValueError("Camera principal point must lie inside the image.")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "image_path", image_path)

    def project_scene_points(
        self,
        points_scene: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project scene points, returning pixel coordinates and camera-Z depth."""
        points_camera = self.camera_to_scene.inverse().apply(points_scene)
        depth = points_camera[..., 2]
        if np.any(depth <= 0.0):
            raise ValueError("All projected points must have positive camera-Z depth.")
        intrinsic = np.asarray(self.intrinsics, dtype=np.float64).reshape(3, 3)
        homogeneous = points_camera @ intrinsic.T
        pixels = homogeneous[..., :2] / homogeneous[..., 2:3]
        if not np.isfinite(pixels).all():
            raise ValueError("Projected pixel coordinates must be finite.")
        return pixels, depth

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON camera representation."""
        return {
            "camera_id": self.camera_id,
            "source_frame_index": self.source_frame_index,
            "width": self.width,
            "height": self.height,
            "intrinsics": list(self.intrinsics),
            "camera_to_scene": self.camera_to_scene.to_list(),
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict camera record and reject unknown keys."""
        raw = _strict_mapping(
            value,
            name="camera",
            keys={
                "camera_id",
                "source_frame_index",
                "width",
                "height",
                "intrinsics",
                "camera_to_scene",
                "image_path",
            },
        )
        intrinsics = raw["intrinsics"]
        transform = raw["camera_to_scene"]
        if not isinstance(intrinsics, Sequence) or isinstance(intrinsics, (str, bytes)):
            raise TypeError("camera.intrinsics must be a sequence.")
        if not isinstance(transform, Sequence) or isinstance(transform, (str, bytes)):
            raise TypeError("camera.camera_to_scene must be a sequence.")
        return cls(
            camera_id=_string(raw["camera_id"], name="camera_id"),
            source_frame_index=_integer(raw["source_frame_index"], name="source_frame_index"),
            width=_integer(raw["width"], name="width", minimum=2),
            height=_integer(raw["height"], name="height", minimum=2),
            intrinsics=_finite_tuple(intrinsics, size=9, name="intrinsics"),
            camera_to_scene=RigidTransform(_finite_tuple(transform, size=16, name="camera_to_scene")),
            image_path=_string(raw["image_path"], name="image_path"),
        )


@dataclass(frozen=True, slots=True)
class CourtInstance:
    """One accepted court and its complete fit/holdout evidence."""

    court_instance_id: str
    candidate_id: str
    scene_from_court: RigidTransform
    court_from_scene: RigidTransform
    fit_status: str
    fit_metrics: Mapping[str, object]
    holdout_status: str
    holdout_metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_id(self.court_instance_id, name="court_instance_id")
        _validate_id(self.candidate_id, name="candidate_id")
        fit_status = _string(self.fit_status, name="fit_status")
        holdout_status = _string(self.holdout_status, name="holdout_status")
        if fit_status != "accepted" or holdout_status != "accepted":
            raise ValueError("Only fit- and holdout-accepted courts may enter MultiCourtLayout.")
        product = self.court_from_scene.matrix() @ self.scene_from_court.matrix()
        if not np.allclose(product, np.eye(4), atol=_MATRIX_ATOL, rtol=0.0):
            raise ValueError("court_from_scene and scene_from_court must be reciprocal.")
        fit_metrics = _json_value(self.fit_metrics, name="fit_metrics")
        holdout_metrics = _json_value(self.holdout_metrics, name="holdout_metrics")
        if not isinstance(fit_metrics, dict) or not isinstance(holdout_metrics, dict):
            raise TypeError("Court metrics must be mappings.")
        object.__setattr__(self, "fit_status", fit_status)
        object.__setattr__(self, "holdout_status", holdout_status)
        object.__setattr__(self, "fit_metrics", fit_metrics)
        object.__setattr__(self, "holdout_metrics", holdout_metrics)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical court record."""
        return {
            "court_instance_id": self.court_instance_id,
            "candidate_id": self.candidate_id,
            "scene_from_court": self.scene_from_court.to_list(),
            "court_from_scene": self.court_from_scene.to_list(),
            "fit_status": self.fit_status,
            "fit_metrics": dict(self.fit_metrics),
            "holdout_status": self.holdout_status,
            "holdout_metrics": dict(self.holdout_metrics),
        }


@dataclass(frozen=True, slots=True)
class MultiCourtLayout:
    """All accepted courts in one scene and their scene-space complex bounds."""

    courts: tuple[CourtInstance, ...]
    complex_bounds_scene: tuple[float, ...]
    primary_court_instance_id: str | None

    def __post_init__(self) -> None:
        if not self.courts:
            raise ValueError("MultiCourtLayout must contain at least one accepted court.")
        court_ids = [court.court_instance_id for court in self.courts]
        candidate_ids = [court.candidate_id for court in self.courts]
        if len(court_ids) != len(set(court_ids)):
            raise ValueError("court_instance_id values must be unique.")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique.")
        bounds = _finite_tuple(self.complex_bounds_scene, size=6, name="complex_bounds_scene")
        bounds_array = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        if np.any(bounds_array[0] >= bounds_array[1]):
            raise ValueError("complex bounds minimum must be strictly below maximum.")
        if self.primary_court_instance_id is not None:
            _validate_id(self.primary_court_instance_id, name="primary_court_instance_id")
            if self.primary_court_instance_id not in court_ids:
                raise ValueError("primary_court_instance_id must reference an accepted court.")
        object.__setattr__(self, "complex_bounds_scene", bounds)

    def court(self, court_instance_id: str) -> CourtInstance:
        """Return one court by ID or fail rather than selecting a fallback."""
        matches = [court for court in self.courts if court.court_instance_id == court_instance_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown court_instance_id: {court_instance_id!r}.")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        """Return the strict multi-court layout representation."""
        return {
            "schema": "multi_court_layout_v1",
            "courts": [court.to_dict() for court in self.courts],
            "complex_bounds_scene": list(self.complex_bounds_scene),
            "primary_court_instance_id": self.primary_court_instance_id,
        }


__all__ = [
    "CAMERA_AXES_OPENCV",
    "COURT_AXES_METRES",
    "PIXEL_COORDINATES",
    "SCENE_CONTRACT_SCHEMA",
    "CourtInstance",
    "MultiCourtLayout",
    "RigidTransform",
    "SceneCamera",
]
