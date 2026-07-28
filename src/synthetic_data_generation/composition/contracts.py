"""Versioned file contracts for composing movable Gaussians into one scene."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

from src.synthetic_data_generation.scene_contract import (
    ArtifactRef,
    SimilarityTransform,
)

GAUSSIAN_ASSET_SCHEMA = "tennis_gaussian_asset_v1"
GAUSSIAN_SCENE_SCHEMA = "tennis_gaussian_scene_composition_v1"
NHT_TENSOR_ENCODING = "nht_raw_parameters_v1"
NHT_APPEARANCE_MODEL = "nht_deferred_v1"
SCENE_COORDINATE_FRAME = "scene"
ASSET_COORDINATE_FRAME = "asset_local"
SCENE_UNIT = "scene_unit"
METRE_UNIT = "metre"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class GaussianAsset:
    """One immutable background or metric-local movable Gaussian asset."""

    schema: str
    asset_id: str
    asset_class: str
    role: Literal["background", "movable"]
    coordinate_frame: str
    unit: str
    metres_per_unit: float | None
    gaussian_count: int
    feature_dim: int
    tensor_encoding: str
    tensors: ArtifactRef
    appearance_model: str
    appearance_space_sha256: str
    appearance_payload: ArtifactRef
    provenance: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if self.schema != GAUSSIAN_ASSET_SCHEMA:
            raise ValueError(
                f"Unsupported Gaussian asset schema {self.schema!r}; "
                f"expected {GAUSSIAN_ASSET_SCHEMA!r}."
            )
        _validate_id(self.asset_id, name="asset_id")
        _validate_id(self.asset_class, name="asset_class")
        if self.role not in {"background", "movable"}:
            raise ValueError(f"Unsupported Gaussian asset role: {self.role!r}.")
        if isinstance(self.gaussian_count, bool) or self.gaussian_count <= 0:
            raise ValueError("gaussian_count must be a positive integer.")
        if isinstance(self.feature_dim, bool) or self.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        if self.tensor_encoding != NHT_TENSOR_ENCODING:
            raise ValueError(f"Unsupported tensor encoding: {self.tensor_encoding!r}.")
        if self.appearance_model != NHT_APPEARANCE_MODEL:
            raise ValueError(
                f"Unsupported appearance model: {self.appearance_model!r}."
            )
        appearance_digest = _sha256(
            self.appearance_space_sha256,
            name="appearance_space_sha256",
        )
        provenance = tuple(self.provenance)
        if not provenance:
            raise ValueError("Gaussian asset provenance must not be empty.")
        _require_unique(
            [artifact.artifact_id for artifact in provenance],
            name="Gaussian asset provenance ids",
        )

        if self.role == "background":
            if (
                self.coordinate_frame != SCENE_COORDINATE_FRAME
                or self.unit != SCENE_UNIT
                or self.metres_per_unit is not None
            ):
                raise ValueError(
                    "Background assets must use scene/scene_unit with no "
                    "independent metre conversion."
                )
        else:
            metres_per_unit = _positive_float(
                self.metres_per_unit,
                name="metres_per_unit",
            )
            if (
                self.coordinate_frame != ASSET_COORDINATE_FRAME
                or self.unit != METRE_UNIT
                or metres_per_unit != 1.0
            ):
                raise ValueError(
                    "Movable assets must be canonical asset_local coordinates "
                    "expressed directly in metres."
                )

        object.__setattr__(self, "appearance_space_sha256", appearance_digest)
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible strict asset record."""
        return {
            "schema": self.schema,
            "asset_id": self.asset_id,
            "asset_class": self.asset_class,
            "role": self.role,
            "coordinate_frame": self.coordinate_frame,
            "unit": self.unit,
            "metres_per_unit": self.metres_per_unit,
            "gaussian_count": self.gaussian_count,
            "feature_dim": self.feature_dim,
            "tensor_encoding": self.tensor_encoding,
            "tensors": self.tensors.to_dict(),
            "appearance_model": self.appearance_model,
            "appearance_space_sha256": self.appearance_space_sha256,
            "appearance_payload": self.appearance_payload.to_dict(),
            "provenance": [artifact.to_dict() for artifact in self.provenance],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and validate one exact v1 Gaussian asset record."""
        raw = _strict_mapping(
            value,
            name="Gaussian asset",
            keys={
                "schema",
                "asset_id",
                "asset_class",
                "role",
                "coordinate_frame",
                "unit",
                "metres_per_unit",
                "gaussian_count",
                "feature_dim",
                "tensor_encoding",
                "tensors",
                "appearance_model",
                "appearance_space_sha256",
                "appearance_payload",
                "provenance",
            },
        )
        raw_role = _string(raw["role"], name="role")
        if raw_role not in {"background", "movable"}:
            raise ValueError(f"Unsupported Gaussian asset role: {raw_role!r}.")
        role = cast(Literal["background", "movable"], raw_role)
        raw_metres = raw["metres_per_unit"]
        metres_per_unit = (
            None if raw_metres is None else _number(raw_metres, name="metres_per_unit")
        )
        return cls(
            schema=_string(raw["schema"], name="schema"),
            asset_id=_string(raw["asset_id"], name="asset_id"),
            asset_class=_string(raw["asset_class"], name="asset_class"),
            role=role,
            coordinate_frame=_string(
                raw["coordinate_frame"],
                name="coordinate_frame",
            ),
            unit=_string(raw["unit"], name="unit"),
            metres_per_unit=metres_per_unit,
            gaussian_count=_integer(
                raw["gaussian_count"],
                name="gaussian_count",
            ),
            feature_dim=_integer(raw["feature_dim"], name="feature_dim"),
            tensor_encoding=_string(
                raw["tensor_encoding"],
                name="tensor_encoding",
            ),
            tensors=ArtifactRef.from_dict(raw["tensors"]),
            appearance_model=_string(
                raw["appearance_model"],
                name="appearance_model",
            ),
            appearance_space_sha256=_string(
                raw["appearance_space_sha256"],
                name="appearance_space_sha256",
            ),
            appearance_payload=ArtifactRef.from_dict(raw["appearance_payload"]),
            provenance=_artifact_sequence(raw["provenance"], name="provenance"),
        )


@dataclass(frozen=True)
class GaussianInstance:
    """One movable asset instance placed into scene coordinates."""

    instance_id: int
    asset: GaussianAsset
    scene_from_asset: SimilarityTransform

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or self.instance_id <= 0:
            raise ValueError("Movable instance_id must be a positive integer.")
        if self.asset.role != "movable":
            raise ValueError("Gaussian instances must reference movable assets.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible instance record."""
        return {
            "instance_id": self.instance_id,
            "asset": self.asset.to_dict(),
            "scene_from_asset": self.scene_from_asset.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict movable-instance record."""
        raw = _strict_mapping(
            value,
            name="Gaussian instance",
            keys={"instance_id", "asset", "scene_from_asset"},
        )
        return cls(
            instance_id=_integer(raw["instance_id"], name="instance_id"),
            asset=GaussianAsset.from_dict(raw["asset"]),
            scene_from_asset=SimilarityTransform.from_dict(raw["scene_from_asset"]),
        )


@dataclass(frozen=True)
class GaussianSceneComposition:
    """One immutable all-Gaussian scene submitted to a single renderer call."""

    schema: str
    composition_id: str
    composition_fingerprint: str
    scene_source: ArtifactRef
    background: GaussianAsset
    instances: tuple[GaussianInstance, ...]
    renderer_backend: str
    renderer_commit: str

    def __post_init__(self) -> None:
        if self.schema != GAUSSIAN_SCENE_SCHEMA:
            raise ValueError(
                f"Unsupported Gaussian scene schema {self.schema!r}; "
                f"expected {GAUSSIAN_SCENE_SCHEMA!r}."
            )
        _validate_id(self.composition_id, name="composition_id")
        if self.background.role != "background":
            raise ValueError("Scene background must reference a background asset.")
        instances = tuple(self.instances)
        if not instances:
            raise ValueError("Gaussian composition requires at least one instance.")
        _require_unique(
            [instance.instance_id for instance in instances],
            name="Gaussian instance ids",
        )
        if not self.renderer_backend.strip():
            raise ValueError("renderer_backend must not be empty.")
        renderer_commit = self.renderer_commit.lower()
        if _GIT_COMMIT_PATTERN.fullmatch(renderer_commit) is None:
            raise ValueError(
                f"renderer_commit must be a full Git commit: {self.renderer_commit!r}."
            )

        appearance_spaces = {
            self.background.appearance_space_sha256,
            *(instance.asset.appearance_space_sha256 for instance in instances),
        }
        if len(appearance_spaces) != 1:
            raise ValueError(
                "All composed NHT assets must share one exact appearance space; "
                "independently trained deferred features cannot be concatenated."
            )
        if any(
            instance.asset.feature_dim != self.background.feature_dim
            for instance in instances
        ):
            raise ValueError("All composed assets must have the same feature_dim.")

        fingerprint = _sha256(
            self.composition_fingerprint,
            name="composition_fingerprint",
        )
        expected = compute_composition_fingerprint(
            composition_id=self.composition_id,
            scene_source=self.scene_source,
            background=self.background,
            instances=instances,
            renderer_backend=self.renderer_backend,
            renderer_commit=renderer_commit,
        )
        if fingerprint != expected:
            raise ValueError(
                "Gaussian composition fingerprint mismatch: "
                f"declared {fingerprint}, computed {expected}."
            )
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "renderer_commit", renderer_commit)
        object.__setattr__(self, "composition_fingerprint", fingerprint)

    @classmethod
    def create(
        cls,
        *,
        composition_id: str,
        scene_source: ArtifactRef,
        background: GaussianAsset,
        instances: Sequence[GaussianInstance],
        renderer_backend: str,
        renderer_commit: str,
    ) -> Self:
        """Create a strict composition with a canonical content fingerprint."""
        instance_tuple = tuple(instances)
        commit = renderer_commit.lower()
        return cls(
            schema=GAUSSIAN_SCENE_SCHEMA,
            composition_id=composition_id,
            composition_fingerprint=compute_composition_fingerprint(
                composition_id=composition_id,
                scene_source=scene_source,
                background=background,
                instances=instance_tuple,
                renderer_backend=renderer_backend,
                renderer_commit=commit,
            ),
            scene_source=scene_source,
            background=background,
            instances=instance_tuple,
            renderer_backend=renderer_backend,
            renderer_commit=commit,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible composition record."""
        return {
            "schema": self.schema,
            "composition_id": self.composition_id,
            "composition_fingerprint": self.composition_fingerprint,
            "scene_source": self.scene_source.to_dict(),
            "background": self.background.to_dict(),
            "instances": [instance.to_dict() for instance in self.instances],
            "renderer_backend": self.renderer_backend,
            "renderer_commit": self.renderer_commit,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and verify one exact v1 composition record."""
        raw = _strict_mapping(
            value,
            name="Gaussian scene composition",
            keys={
                "schema",
                "composition_id",
                "composition_fingerprint",
                "scene_source",
                "background",
                "instances",
                "renderer_backend",
                "renderer_commit",
            },
        )
        return cls(
            schema=_string(raw["schema"], name="schema"),
            composition_id=_string(
                raw["composition_id"],
                name="composition_id",
            ),
            composition_fingerprint=_string(
                raw["composition_fingerprint"],
                name="composition_fingerprint",
            ),
            scene_source=ArtifactRef.from_dict(raw["scene_source"]),
            background=GaussianAsset.from_dict(raw["background"]),
            instances=_instance_sequence(raw["instances"]),
            renderer_backend=_string(
                raw["renderer_backend"],
                name="renderer_backend",
            ),
            renderer_commit=_string(
                raw["renderer_commit"],
                name="renderer_commit",
            ),
        )


def compute_composition_fingerprint(
    *,
    composition_id: str,
    scene_source: ArtifactRef,
    background: GaussianAsset,
    instances: Sequence[GaussianInstance],
    renderer_backend: str,
    renderer_commit: str,
) -> str:
    """Hash every input which can change Gaussian scene composition."""
    payload = {
        "schema": GAUSSIAN_SCENE_SCHEMA,
        "composition_id": composition_id,
        "scene_source": artifact_content_identity(scene_source),
        "background": gaussian_asset_content_identity(background),
        "instances": [_instance_content_identity(instance) for instance in instances],
        "renderer_backend": renderer_backend,
        "renderer_commit": renderer_commit,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_gaussian_scene_manifest(
    path: Path,
    composition: GaussianSceneComposition,
) -> None:
    """Atomically publish a composition manifest without replacing an artifact."""
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite Gaussian composition: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(composition.to_dict(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_gaussian_scene_manifest(path: Path) -> GaussianSceneComposition:
    """Load and strictly validate one composition manifest."""
    with path.resolve().open(encoding="utf-8") as handle:
        return GaussianSceneComposition.from_dict(json.load(handle))


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    raw = {str(key): item for key, item in value.items()}
    missing = keys.difference(raw)
    extra = set(raw).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys differ; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return raw


def artifact_content_identity(artifact: ArtifactRef) -> dict[str, object]:
    """Return path-independent content identity for an artifact reference."""
    return {
        "artifact_id": artifact.artifact_id,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def gaussian_asset_content_identity(asset: GaussianAsset) -> dict[str, object]:
    """Return path-independent content identity for one Gaussian asset."""
    payload = asset.to_dict()
    payload["tensors"] = artifact_content_identity(asset.tensors)
    payload["appearance_payload"] = artifact_content_identity(asset.appearance_payload)
    payload["provenance"] = [
        artifact_content_identity(artifact) for artifact in asset.provenance
    ]
    return payload


def _instance_content_identity(instance: GaussianInstance) -> dict[str, object]:
    return {
        "instance_id": instance.instance_id,
        "asset": gaussian_asset_content_identity(instance.asset),
        "scene_from_asset": instance.scene_from_asset.to_dict(),
    }


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: object, *, name: str) -> float:
    result = _number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _sha256(value: str, *, name: str) -> str:
    digest = value.lower()
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a full SHA-256 digest.")
    return digest


def _validate_id(value: str, *, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid {name}: {value!r}.")


def _require_unique(values: Sequence[object], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates.")


def _artifact_sequence(value: object, *, name: str) -> tuple[ArtifactRef, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return tuple(ArtifactRef.from_dict(item) for item in value)


def _instance_sequence(value: object) -> tuple[GaussianInstance, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("instances must be a JSON array.")
    return tuple(GaussianInstance.from_dict(item) for item in value)
