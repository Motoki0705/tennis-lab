"""Versioned scene/camera contract for external rendering backends.

The contract contains only plain geometry, artifact provenance, and an accepted
court-to-scene alignment. It intentionally has no dependency on BLCS, ball
detection, gsplat, or a renderer implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

SCENE_CONTRACT_SCHEMA = "tennis_scene_contract_v1"
CAMERA_AXES_OPENCV = "opencv:+x_right,+y_down,+z_forward"
COURT_AXES_METRES = "right_handed_metres:+x_right_sideline,+y_far_baseline,+z_up"
PIXEL_COORDINATES = "undistorted_cropped_zero_based_pixel_centres"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MATRIX_ATOL = 1.0e-6


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed artifact reference independent of provider layout."""

    artifact_id: str
    uri: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_id(self.artifact_id, name="artifact_id")
        if not self.uri.strip():
            raise ValueError("Artifact uri must not be empty.")
        digest = self.sha256.lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"Invalid SHA-256 digest: {self.sha256!r}.")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("Artifact size_bytes must be a non-negative integer.")
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict artifact record."""
        raw = _strict_mapping(
            value,
            name="artifact",
            keys={"artifact_id", "uri", "sha256", "size_bytes"},
        )
        return cls(
            artifact_id=_string(raw["artifact_id"], name="artifact_id"),
            uri=_string(raw["uri"], name="uri"),
            sha256=_string(raw["sha256"], name="sha256"),
            size_bytes=_integer(raw["size_bytes"], name="size_bytes"),
        )


@dataclass(frozen=True)
class SimilarityTransform:
    """A positive-scale, proper-rotation Sim(3) acting on column vectors."""

    scale: float
    rotation: tuple[float, ...]
    translation: tuple[float, ...]

    def __post_init__(self) -> None:
        scale = _finite_float(self.scale, name="scale")
        if scale <= 0.0:
            raise ValueError(f"Similarity scale must be positive, got {scale}.")
        rotation = _float_tuple(self.rotation, length=9, name="rotation")
        translation = _float_tuple(
            self.translation,
            length=3,
            name="translation",
        )
        matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        _validate_rotation(matrix, name="rotation")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def matrix(self) -> NDArray[np.float64]:
        """Return the homogeneous 4x4 matrix ``x_out = s R x_in + t``."""
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.scale * np.asarray(
            self.rotation,
            dtype=np.float64,
        ).reshape(3, 3)
        result[:3, 3] = np.asarray(self.translation, dtype=np.float64)
        return result

    def inverse(self) -> SimilarityTransform:
        """Return the exact inverse similarity."""
        rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(self.translation, dtype=np.float64)
        inverse_scale = 1.0 / self.scale
        inverse_rotation = rotation.T
        inverse_translation = -inverse_scale * inverse_rotation @ translation
        return SimilarityTransform(
            scale=inverse_scale,
            rotation=tuple(float(value) for value in inverse_rotation.ravel()),
            translation=tuple(float(value) for value in inverse_translation),
        )

    def apply(self, points: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Apply the transform to an ``(..., 3)`` point array."""
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != 3:
            raise ValueError(f"points must have shape (..., 3), got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("points must contain only finite values.")
        rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(self.translation, dtype=np.float64)
        return self.scale * (array @ rotation.T) + translation

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "scale": self.scale,
            "rotation": list(self.rotation),
            "translation": list(self.translation),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict similarity-transform record."""
        raw = _strict_mapping(
            value,
            name="similarity",
            keys={"scale", "rotation", "translation"},
        )
        return cls(
            scale=_number(raw["scale"], name="scale"),
            rotation=_number_sequence(
                raw["rotation"],
                length=9,
                name="rotation",
            ),
            translation=_number_sequence(
                raw["translation"],
                length=3,
                name="translation",
            ),
        )


@dataclass(frozen=True)
class SceneCamera:
    """One accepted captured pose in the renderer's exact pixel coordinates."""

    camera_id: str
    source_camera_id: str
    image_uri: str
    source_frame_index: int
    group_id: int
    width: int
    height: int
    intrinsics: tuple[float, ...]
    camera_to_scene: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_id(self.camera_id, name="camera_id")
        if not self.source_camera_id.strip():
            raise ValueError("source_camera_id must not be empty.")
        if not self.image_uri.strip():
            raise ValueError("image_uri must not be empty.")
        if isinstance(self.source_frame_index, bool) or self.source_frame_index < 0:
            raise ValueError("source_frame_index must be a non-negative integer.")
        if isinstance(self.group_id, bool) or self.group_id < 0:
            raise ValueError("group_id must be a non-negative integer.")
        if isinstance(self.width, bool) or self.width <= 1:
            raise ValueError("Camera width must be greater than one.")
        if isinstance(self.height, bool) or self.height <= 1:
            raise ValueError("Camera height must be greater than one.")

        intrinsics = _float_tuple(
            self.intrinsics,
            length=9,
            name="intrinsics",
        )
        camera_to_scene = _float_tuple(
            self.camera_to_scene,
            length=16,
            name="camera_to_scene",
        )
        intrinsic_matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        if intrinsic_matrix[0, 0] <= 0.0 or intrinsic_matrix[1, 1] <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")
        if not np.allclose(
            intrinsic_matrix[2],
            np.asarray([0.0, 0.0, 1.0]),
            atol=_MATRIX_ATOL,
            rtol=0.0,
        ):
            raise ValueError("Camera intrinsics must have bottom row [0, 0, 1].")
        if not 0.0 <= intrinsic_matrix[0, 2] < self.width:
            raise ValueError("Camera principal point cx must lie inside the image.")
        if not 0.0 <= intrinsic_matrix[1, 2] < self.height:
            raise ValueError("Camera principal point cy must lie inside the image.")

        pose = np.asarray(camera_to_scene, dtype=np.float64).reshape(4, 4)
        if not np.allclose(
            pose[3],
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            atol=_MATRIX_ATOL,
            rtol=0.0,
        ):
            raise ValueError("camera_to_scene must have bottom row [0, 0, 0, 1].")
        _validate_rotation(pose[:3, :3], name="camera_to_scene rotation")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_scene", camera_to_scene)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "camera_id": self.camera_id,
            "source_camera_id": self.source_camera_id,
            "image_uri": self.image_uri,
            "source_frame_index": self.source_frame_index,
            "group_id": self.group_id,
            "width": self.width,
            "height": self.height,
            "intrinsics": list(self.intrinsics),
            "camera_to_scene": list(self.camera_to_scene),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict camera record."""
        raw = _strict_mapping(
            value,
            name="camera",
            keys={
                "camera_id",
                "source_camera_id",
                "image_uri",
                "source_frame_index",
                "group_id",
                "width",
                "height",
                "intrinsics",
                "camera_to_scene",
            },
        )
        return cls(
            camera_id=_string(raw["camera_id"], name="camera_id"),
            source_camera_id=_string(
                raw["source_camera_id"],
                name="source_camera_id",
            ),
            image_uri=_string(raw["image_uri"], name="image_uri"),
            source_frame_index=_integer(
                raw["source_frame_index"],
                name="source_frame_index",
            ),
            group_id=_integer(raw["group_id"], name="group_id"),
            width=_integer(raw["width"], name="width"),
            height=_integer(raw["height"], name="height"),
            intrinsics=_number_sequence(
                raw["intrinsics"],
                length=9,
                name="intrinsics",
            ),
            camera_to_scene=_number_sequence(
                raw["camera_to_scene"],
                length=16,
                name="camera_to_scene",
            ),
        )


@dataclass(frozen=True)
class AcceptedAlignment:
    """An alignment accepted by the referenced immutable decision manifest."""

    alignment_id: str
    accepted: bool
    selected_court_cluster: str
    selected_symmetry: str
    fit_camera_ids: tuple[str, ...]
    holdout_camera_ids: tuple[str, ...]
    scene_from_court: SimilarityTransform
    court_from_scene: SimilarityTransform
    manifest: ArtifactRef

    def __post_init__(self) -> None:
        _validate_id(self.alignment_id, name="alignment_id")
        if self.accepted is not True:
            raise ValueError("Scene contracts require an accepted alignment.")
        _validate_id(self.selected_court_cluster, name="selected_court_cluster")
        _validate_id(self.selected_symmetry, name="selected_symmetry")
        fit_camera_ids = tuple(self.fit_camera_ids)
        holdout_camera_ids = tuple(self.holdout_camera_ids)
        if not fit_camera_ids or not holdout_camera_ids:
            raise ValueError("Alignment fit and holdout camera ids must be non-empty.")
        if len(set(fit_camera_ids)) != len(fit_camera_ids):
            raise ValueError("fit_camera_ids contains duplicates.")
        if len(set(holdout_camera_ids)) != len(holdout_camera_ids):
            raise ValueError("holdout_camera_ids contains duplicates.")
        overlap = set(fit_camera_ids).intersection(holdout_camera_ids)
        if overlap:
            raise ValueError(
                f"Alignment fit and holdout camera ids overlap: {sorted(overlap)}."
            )
        for camera_id in fit_camera_ids + holdout_camera_ids:
            _validate_id(camera_id, name="alignment camera id")

        scene_from_court = self.scene_from_court.matrix()
        court_from_scene = self.court_from_scene.matrix()
        if np.allclose(
            scene_from_court,
            np.eye(4, dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("An identity court alignment is not an accepted fallback.")
        inverse_error = float(
            np.max(
                np.abs(
                    court_from_scene @ scene_from_court - np.eye(4, dtype=np.float64)
                )
            )
        )
        if inverse_error > _MATRIX_ATOL:
            raise ValueError(
                "scene_from_court and court_from_scene are inconsistent: "
                f"max error {inverse_error:.3g}."
            )
        object.__setattr__(self, "fit_camera_ids", fit_camera_ids)
        object.__setattr__(self, "holdout_camera_ids", holdout_camera_ids)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "alignment_id": self.alignment_id,
            "accepted": self.accepted,
            "selected_court_cluster": self.selected_court_cluster,
            "selected_symmetry": self.selected_symmetry,
            "fit_camera_ids": list(self.fit_camera_ids),
            "holdout_camera_ids": list(self.holdout_camera_ids),
            "scene_from_court": self.scene_from_court.to_dict(),
            "court_from_scene": self.court_from_scene.to_dict(),
            "manifest": self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict accepted-alignment record."""
        raw = _strict_mapping(
            value,
            name="alignment",
            keys={
                "alignment_id",
                "accepted",
                "selected_court_cluster",
                "selected_symmetry",
                "fit_camera_ids",
                "holdout_camera_ids",
                "scene_from_court",
                "court_from_scene",
                "manifest",
            },
        )
        return cls(
            alignment_id=_string(raw["alignment_id"], name="alignment_id"),
            accepted=_boolean(raw["accepted"], name="accepted"),
            selected_court_cluster=_string(
                raw["selected_court_cluster"],
                name="selected_court_cluster",
            ),
            selected_symmetry=_string(
                raw["selected_symmetry"],
                name="selected_symmetry",
            ),
            fit_camera_ids=_string_sequence(
                raw["fit_camera_ids"],
                name="fit_camera_ids",
            ),
            holdout_camera_ids=_string_sequence(
                raw["holdout_camera_ids"],
                name="holdout_camera_ids",
            ),
            scene_from_court=SimilarityTransform.from_dict(raw["scene_from_court"]),
            court_from_scene=SimilarityTransform.from_dict(raw["court_from_scene"]),
            manifest=ArtifactRef.from_dict(raw["manifest"]),
        )


@dataclass(frozen=True)
class SceneContract:
    """Complete accepted scene contract consumed by renderer-independent code."""

    schema: str
    scene_id: str
    scene_fingerprint: str
    provider_backend: str
    camera_axes: str
    court_axes: str
    pixel_coordinates: str
    artifacts: tuple[ArtifactRef, ...]
    cameras: tuple[SceneCamera, ...]
    alignment: AcceptedAlignment

    def __post_init__(self) -> None:
        if self.schema != SCENE_CONTRACT_SCHEMA:
            raise ValueError(
                f"Unsupported scene contract schema {self.schema!r}; "
                f"expected {SCENE_CONTRACT_SCHEMA!r}."
            )
        _validate_id(self.scene_id, name="scene_id")
        if not self.provider_backend.strip():
            raise ValueError("provider_backend must not be empty.")
        if self.camera_axes != CAMERA_AXES_OPENCV:
            raise ValueError(f"Unsupported camera axes: {self.camera_axes!r}.")
        if self.court_axes != COURT_AXES_METRES:
            raise ValueError(f"Unsupported court axes: {self.court_axes!r}.")
        if self.pixel_coordinates != PIXEL_COORDINATES:
            raise ValueError(
                f"Unsupported pixel coordinates: {self.pixel_coordinates!r}."
            )

        artifacts = tuple(self.artifacts)
        cameras = tuple(self.cameras)
        if not artifacts:
            raise ValueError("Scene contract must reference at least one artifact.")
        if not cameras:
            raise ValueError("Scene contract must contain at least one camera.")
        _require_unique(
            [artifact.artifact_id for artifact in artifacts],
            name="artifact ids",
        )
        camera_ids = [camera.camera_id for camera in cameras]
        _require_unique(camera_ids, name="camera ids")
        unknown_alignment_ids = (
            set(self.alignment.fit_camera_ids)
            .union(self.alignment.holdout_camera_ids)
            .difference(camera_ids)
        )
        if unknown_alignment_ids:
            raise ValueError(
                "Alignment references unknown camera ids: "
                f"{sorted(unknown_alignment_ids)}."
            )

        digest = self.scene_fingerprint.lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"Invalid scene fingerprint: {self.scene_fingerprint!r}.")
        expected = compute_scene_fingerprint(
            provider_backend=self.provider_backend,
            artifacts=artifacts,
            cameras=cameras,
        )
        if digest != expected:
            raise ValueError(
                f"Scene fingerprint mismatch: declared {digest}, computed {expected}."
            )
        object.__setattr__(self, "scene_fingerprint", digest)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "cameras", cameras)

    @classmethod
    def create(
        cls,
        *,
        scene_id: str,
        provider_backend: str,
        artifacts: Sequence[ArtifactRef],
        cameras: Sequence[SceneCamera],
        alignment: AcceptedAlignment,
    ) -> Self:
        """Create a v1 contract with its canonical scene fingerprint."""
        artifact_tuple = tuple(artifacts)
        camera_tuple = tuple(cameras)
        return cls(
            schema=SCENE_CONTRACT_SCHEMA,
            scene_id=scene_id,
            scene_fingerprint=compute_scene_fingerprint(
                provider_backend=provider_backend,
                artifacts=artifact_tuple,
                cameras=camera_tuple,
            ),
            provider_backend=provider_backend,
            camera_axes=CAMERA_AXES_OPENCV,
            court_axes=COURT_AXES_METRES,
            pixel_coordinates=PIXEL_COORDINATES,
            artifacts=artifact_tuple,
            cameras=camera_tuple,
            alignment=alignment,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "scene_fingerprint": self.scene_fingerprint,
            "provider_backend": self.provider_backend,
            "camera_axes": self.camera_axes,
            "court_axes": self.court_axes,
            "pixel_coordinates": self.pixel_coordinates,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "alignment": self.alignment.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and validate one strict v1 scene contract."""
        raw = _strict_mapping(
            value,
            name="scene contract",
            keys={
                "schema",
                "scene_id",
                "scene_fingerprint",
                "provider_backend",
                "camera_axes",
                "court_axes",
                "pixel_coordinates",
                "artifacts",
                "cameras",
                "alignment",
            },
        )
        return cls(
            schema=_string(raw["schema"], name="schema"),
            scene_id=_string(raw["scene_id"], name="scene_id"),
            scene_fingerprint=_string(
                raw["scene_fingerprint"],
                name="scene_fingerprint",
            ),
            provider_backend=_string(
                raw["provider_backend"],
                name="provider_backend",
            ),
            camera_axes=_string(raw["camera_axes"], name="camera_axes"),
            court_axes=_string(raw["court_axes"], name="court_axes"),
            pixel_coordinates=_string(
                raw["pixel_coordinates"],
                name="pixel_coordinates",
            ),
            artifacts=tuple(
                ArtifactRef.from_dict(item)
                for item in _sequence(raw["artifacts"], name="artifacts")
            ),
            cameras=tuple(
                SceneCamera.from_dict(item)
                for item in _sequence(raw["cameras"], name="cameras")
            ),
            alignment=AcceptedAlignment.from_dict(raw["alignment"]),
        )


def compute_scene_fingerprint(
    *,
    provider_backend: str,
    artifacts: Sequence[ArtifactRef],
    cameras: Sequence[SceneCamera],
) -> str:
    """Hash immutable provider artifacts and calibrated captured cameras."""
    if not provider_backend.strip():
        raise ValueError("provider_backend must not be empty.")
    payload = {
        "schema": SCENE_CONTRACT_SCHEMA,
        "provider_backend": provider_backend,
        "camera_axes": CAMERA_AXES_OPENCV,
        "pixel_coordinates": PIXEL_COORDINATES,
        "artifacts": [
            artifact.to_dict()
            for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
        ],
        "cameras": [
            camera.to_dict()
            for camera in sorted(cameras, key=lambda item: item.camera_id)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_scene_contract(path: str | Path) -> SceneContract:
    """Load and strictly validate a scene contract JSON file."""
    contract_path = Path(path)
    with contract_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return SceneContract.from_dict(raw)


def write_scene_contract(
    path: str | Path,
    contract: SceneContract,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a contract, refusing overwrite by default."""
    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        contract.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=contract_path.parent,
            prefix=f".{contract_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, contract_path)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, contract_path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite scene contract: {contract_path}"
                ) from exc
            temporary_path.unlink()
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_rotation(matrix: NDArray[np.float64], *, name: str) -> None:
    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3, dtype=np.float64),
        atol=_MATRIX_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must be orthonormal.")
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.999:
        raise ValueError(
            f"{name} must be a proper rotation with det(R) > 0.999, got {determinant}."
        )


def _float_tuple(
    value: Sequence[object],
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        actual = len(value) if not isinstance(value, (str, bytes)) else "string"
        raise ValueError(f"{name} must contain {length} numbers, got {actual}.")
    result = tuple(_finite_float(item, name=f"{name} item") for item in value)
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    raw = {str(key): item for key, item in value.items()}
    missing = keys.difference(raw)
    extra = set(raw).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return raw


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _number_sequence(
    value: object,
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    sequence = _sequence(value, name=name)
    return _float_tuple(sequence, length=length, name=name)


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(
        _string(item, name=f"{name} item") for item in _sequence(value, name=name)
    )


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    return _finite_float(value, name=name)


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _validate_id(value: str, *, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is not a path-safe identifier: {value!r}.")


def _require_unique(values: Sequence[str], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique.")


__all__ = [
    "AcceptedAlignment",
    "ArtifactRef",
    "CAMERA_AXES_OPENCV",
    "COURT_AXES_METRES",
    "PIXEL_COORDINATES",
    "SCENE_CONTRACT_SCHEMA",
    "SceneCamera",
    "SceneContract",
    "SimilarityTransform",
    "compute_scene_fingerprint",
    "load_scene_contract",
    "write_scene_contract",
]
