"""Source-independent COCO-17 motion contract for PLCS."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

COCO17_MOTION_SCHEMA_VERSION = "coco17_motion_v1"

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


class MotionSourceKind(StrEnum):
    """Motion producers currently supported by the canonical adapter layer."""

    ACCAD = "accad"
    GVHMR = "gvhmr"


def _trimmed(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _readonly_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int | str, ...],
) -> NDArray[Any]:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {array.dtype}.")
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    for actual, expected in zip(array.shape, shape, strict=True):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _immutable_json_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("provenance must be a mapping with string keys.")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError("provenance must contain only finite JSON values.") from error
    if not isinstance(decoded, dict):
        raise TypeError("provenance must encode as a JSON object.")
    return MappingProxyType(cast(dict[str, object], decoded))


@dataclass(frozen=True, slots=True)
class Coco17MotionClip:
    """One losslessly timed motion in a right-handed, metre, Z-up world frame.

    ``joints_3d_m`` follows the canonical COCO-17 order from
    :mod:`src.utils.schema.player`. Root translation and full root rotation are
    separate required signals: COCO-17 has no pelvis joint and must not be used
    to silently reconstruct either signal.
    """

    source_id: str
    source_path: str
    source_kind: MotionSourceKind
    category: str
    gender: str
    fps: float
    timestamps_s: Float64Array
    joints_3d_m: Float32Array
    root_translation_m: Float32Array
    root_rotation: Float32Array
    joint_confidence: Float32Array
    frame_valid: BoolArray
    provenance: Mapping[str, object]
    frame_count: int = field(init=False)

    def __post_init__(self) -> None:
        source_id = _trimmed(self.source_id, name="source_id")
        source_path = _trimmed(self.source_path, name="source_path")
        category = _trimmed(self.category, name="category")
        gender = _trimmed(self.gender, name="gender").lower()
        if gender not in {"female", "male", "neutral"}:
            raise ValueError("gender must be female, male, or neutral.")
        try:
            source_kind = MotionSourceKind(self.source_kind)
        except ValueError as error:
            raise ValueError("source_kind must be accad or gvhmr.") from error
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise TypeError("fps must be numeric.")
        fps = float(self.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive.")

        timestamps = cast(
            Float64Array,
            _readonly_array(
                self.timestamps_s,
                name="timestamps_s",
                dtype=np.dtype(np.float64),
                shape=("T",),
            ),
        )
        frame_count = int(timestamps.shape[0])
        if frame_count == 0:
            raise ValueError("A motion clip must contain at least one frame.")
        if frame_count > 1 and not bool(np.all(np.diff(timestamps) > 0.0)):
            raise ValueError("timestamps_s must be strictly increasing.")
        if abs(float(timestamps[0])) > 1e-9:
            raise ValueError("timestamps_s must start at zero.")
        if frame_count > 1 and not np.allclose(
            np.diff(timestamps),
            1.0 / fps,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError("timestamps_s must use the constant interval 1 / fps.")

        joints = cast(
            Float32Array,
            _readonly_array(
                self.joints_3d_m,
                name="joints_3d_m",
                dtype=np.dtype(np.float32),
                shape=(frame_count, 17, 3),
            ),
        )
        translation = cast(
            Float32Array,
            _readonly_array(
                self.root_translation_m,
                name="root_translation_m",
                dtype=np.dtype(np.float32),
                shape=(frame_count, 3),
            ),
        )
        rotation = cast(
            Float32Array,
            _readonly_array(
                self.root_rotation,
                name="root_rotation",
                dtype=np.dtype(np.float32),
                shape=(frame_count, 3, 3),
            ),
        )
        confidence = cast(
            Float32Array,
            _readonly_array(
                self.joint_confidence,
                name="joint_confidence",
                dtype=np.dtype(np.float32),
                shape=(frame_count, 17),
            ),
        )
        valid = cast(
            BoolArray,
            _readonly_array(
                self.frame_valid,
                name="frame_valid",
                dtype=np.dtype(np.bool_),
                shape=(frame_count,),
            ),
        )
        if bool(((confidence < 0.0) | (confidence > 1.0)).any()):
            raise ValueError("joint_confidence must be within [0, 1].")

        identities = np.einsum("tji,tjk->tik", rotation, rotation)
        expected = np.broadcast_to(np.eye(3, dtype=np.float32), identities.shape)
        if not np.allclose(identities, expected, atol=2e-4, rtol=0.0):
            raise ValueError("root_rotation must contain orthonormal matrices.")
        determinants = np.linalg.det(rotation)
        if not np.allclose(determinants, 1.0, atol=2e-4, rtol=0.0):
            raise ValueError("root_rotation must contain proper rotations (det=1).")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "gender", gender)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "joints_3d_m", joints)
        object.__setattr__(self, "root_translation_m", translation)
        object.__setattr__(self, "root_rotation", rotation)
        object.__setattr__(self, "joint_confidence", confidence)
        object.__setattr__(self, "frame_valid", valid)
        object.__setattr__(self, "provenance", _immutable_json_mapping(self.provenance))
        object.__setattr__(self, "frame_count", frame_count)

    def metadata(self) -> dict[str, object]:
        """Return the serializable, source-independent metadata document."""
        return {
            "schema_version": COCO17_MOTION_SCHEMA_VERSION,
            "coordinate_system": "right_handed_z_up_m",
            "keypoint_schema": "coco17",
            "source_id": self.source_id,
            "source_path": self.source_path,
            "source_kind": self.source_kind.value,
            "category": self.category,
            "gender": self.gender,
            "native_fps": self.fps,
            "frame_count": self.frame_count,
            "provenance": dict(self.provenance),
        }


__all__ = [
    "COCO17_MOTION_SCHEMA_VERSION",
    "BoolArray",
    "Coco17MotionClip",
    "Float32Array",
    "Float64Array",
    "MotionSourceKind",
]
