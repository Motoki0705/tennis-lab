"""Strict contracts for deterministic synthetic-data publication bundles."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

PUBLICATION_REQUEST_SCHEMA = "synthetic_publication_request_v1"
PUBLICATION_MANIFEST_SCHEMA = "synthetic_publication_manifest_v1"
PUBLICATION_BUNDLE_SCHEMA = "synthetic_publication_bundle_v1"
PUBLICATION_COORDINATE_CONTRACT = (
    "camera=opencv(x-right,y-down,z-forward);"
    "scene=right-handed-metric-metres(x-right-sideline,y-far-baseline,z-up);"
    "nht-to-metric=MetricSceneAdapter"
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PublicationArtifactName(StrEnum):
    """Fixed complete artifact inventory for one publication bundle."""

    DATASET_COURT = "dataset-court.gif"
    DATASET_BLCS = "dataset-blcs.gif"
    DATASET_PLCS = "dataset-plcs.gif"
    ALIGNMENT_PROGRESSION = "alignment-progression.gif"
    ALIGNMENT_HEATMAP_COURT = "alignment-heatmap-court.png"
    CAPTURED_CAMERA_TRAJECTORY = "captured-camera-trajectory.png"
    BLCS_CAMERA_LAYOUT = "blcs-camera-layout.png"
    PLCS_CAMERA_LAYOUT = "plcs-camera-layout.png"
    CAMERA_LAYOUT_COMPARISON = "camera-layout-comparison.png"
    PUBLICATION_OVERVIEW = "publication-overview.png"


REQUIRED_PUBLICATION_ARTIFACTS = tuple(PublicationArtifactName)


@dataclass(frozen=True, slots=True)
class PublicationDrawingSettings:
    """Explicit media, geometry, and bounded-asset drawing authority."""

    dataset_size: tuple[int, int]
    alignment_size: tuple[int, int]
    figure_size: tuple[int, int]
    overview_size: tuple[int, int]
    gif_duration_ms: int
    frustum_depth_metres: float
    line_width: float
    font_size: int
    history_frames: int
    maximum_artifact_bytes: int
    maximum_bundle_bytes: int

    def __post_init__(self) -> None:
        for name in ("dataset_size", "alignment_size", "figure_size", "overview_size"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
                or any(item < 64 or item > 8192 for item in value)
            ):
                raise ValueError(f"drawing.{name} must be two integers in [64, 8192].")
        for name, minimum, maximum in (
            ("gif_duration_ms", 10, 10_000),
            ("font_size", 6, 96),
            ("history_frames", 0, 120),
            ("maximum_artifact_bytes", 1_024, 100_000_000),
            ("maximum_bundle_bytes", 10_240, 500_000_000),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"drawing.{name} must lie in [{minimum}, {maximum}].")
        for name in ("frustum_depth_metres", "line_width"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"drawing.{name} must be positive and finite.")
        if self.maximum_bundle_bytes < self.maximum_artifact_bytes:
            raise ValueError(
                "drawing.maximum_bundle_bytes must be at least maximum_artifact_bytes."
            )
        object.__setattr__(
            self, "frustum_depth_metres", float(self.frustum_depth_metres)
        )
        object.__setattr__(self, "line_width", float(self.line_width))

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-safe resolved drawing settings."""
        return {
            "dataset_size": list(self.dataset_size),
            "alignment_size": list(self.alignment_size),
            "figure_size": list(self.figure_size),
            "overview_size": list(self.overview_size),
            "gif_duration_ms": self.gif_duration_ms,
            "frustum_depth_metres": self.frustum_depth_metres,
            "line_width": self.line_width,
            "font_size": self.font_size,
            "history_frames": self.history_frames,
            "maximum_artifact_bytes": self.maximum_artifact_bytes,
            "maximum_bundle_bytes": self.maximum_bundle_bytes,
        }


@dataclass(frozen=True, slots=True)
class PublicationRequest:
    """One explicit, fail-closed publication request for a canonical scene."""

    scene_id: str
    scene_root: Path
    output_bundle: Path
    artifact_names: tuple[PublicationArtifactName, ...]
    court_trajectory_id: str
    court_frame_indices: tuple[int, ...]
    blcs_logical_scene_id: str
    blcs_camera_id: str
    blcs_frame_indices: tuple[int, ...]
    blcs_camera_ids: tuple[str, ...]
    plcs_logical_scene_id: str
    plcs_camera_id: str
    plcs_frame_indices: tuple[int, ...]
    plcs_camera_ids: tuple[str, ...]
    captured_camera_ids: tuple[str, ...]
    drawing: PublicationDrawingSettings

    def __post_init__(self) -> None:
        _identifier(self.scene_id, name="scene_id")
        scene_root = _absolute_path(self.scene_root, name="scene_root")
        output_bundle = _absolute_path(self.output_bundle, name="output_bundle")
        if scene_root.name != self.scene_id or scene_root.parent.name != "scenes":
            raise ValueError(
                "scene_root must be the canonical .../scenes/<scene_id> owner."
            )
        if scene_root.is_symlink() or not scene_root.is_dir():
            raise ValueError("scene_root must be an existing ordinary directory.")
        if output_bundle.exists() or output_bundle.is_symlink():
            raise FileExistsError(f"Publication output already exists: {output_bundle}")
        if output_bundle.resolve(strict=False).is_relative_to(scene_root.resolve()):
            raise ValueError("Publication output must stay outside the scene owner.")
        artifacts = tuple(self.artifact_names)
        if artifacts != REQUIRED_PUBLICATION_ARTIFACTS:
            raise ValueError(
                "artifact_names must list the fixed complete publication inventory in order."
            )
        for name in (
            "court_trajectory_id",
            "blcs_logical_scene_id",
            "blcs_camera_id",
            "plcs_logical_scene_id",
            "plcs_camera_id",
        ):
            _identifier(getattr(self, name), name=name)
        for name in ("court_frame_indices", "blcs_frame_indices", "plcs_frame_indices"):
            _frame_indices(getattr(self, name), name=name)
        for name in ("blcs_camera_ids", "plcs_camera_ids", "captured_camera_ids"):
            _identifiers(getattr(self, name), name=name)
        if self.blcs_camera_id not in self.blcs_camera_ids:
            raise ValueError("blcs_camera_id must belong to blcs_camera_ids.")
        if self.plcs_camera_id not in self.plcs_camera_ids:
            raise ValueError("plcs_camera_id must belong to plcs_camera_ids.")
        if not isinstance(self.drawing, PublicationDrawingSettings):
            raise TypeError("drawing must be PublicationDrawingSettings.")
        object.__setattr__(self, "scene_root", scene_root)
        object.__setattr__(self, "output_bundle", output_bundle)
        object.__setattr__(self, "artifact_names", artifacts)

    @property
    def alignment_root(self) -> Path:
        """Return the fixed scene alignment owner."""
        return self.scene_root / "alignment"

    @property
    def reconstruction_scene_json(self) -> Path:
        """Return the fixed standard reconstruction export entry."""
        return self.scene_root / "reconstruction" / "export" / "scene.json"

    def dataset_root(self, domain: str) -> Path:
        """Return one fixed dataset owner without accepting aliases."""
        if domain not in {"court", "blcs", "plcs"}:
            raise ValueError(f"Unsupported publication dataset domain: {domain!r}.")
        return self.scene_root / "datasets" / domain

    def to_resolved_config(self) -> dict[str, object]:
        """Return deterministic semantic config relative to the selected scene/output."""
        return {
            "schema": PUBLICATION_REQUEST_SCHEMA,
            "scene_id": self.scene_id,
            "scene_root": ".",
            "output_bundle": ".",
            "artifact_names": [item.value for item in self.artifact_names],
            "court": {
                "dataset_root": "datasets/court",
                "trajectory_id": self.court_trajectory_id,
                "frame_indices": list(self.court_frame_indices),
            },
            "blcs": {
                "dataset_root": "datasets/blcs",
                "logical_scene_id": self.blcs_logical_scene_id,
                "camera_id": self.blcs_camera_id,
                "frame_indices": list(self.blcs_frame_indices),
                "camera_ids": list(self.blcs_camera_ids),
            },
            "plcs": {
                "dataset_root": "datasets/plcs",
                "logical_scene_id": self.plcs_logical_scene_id,
                "camera_id": self.plcs_camera_id,
                "frame_indices": list(self.plcs_frame_indices),
                "camera_ids": list(self.plcs_camera_ids),
            },
            "captured": {
                "scene_json": "reconstruction/export/scene.json",
                "camera_ids": list(self.captured_camera_ids),
            },
            "alignment_root": "alignment",
            "drawing": self.drawing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PublicationArtifactRecord:
    """One manifest-listed media artifact and its semantic mapping."""

    file_name: PublicationArtifactName
    media_type: str
    width: int
    height: int
    frame_count: int
    duration_ms: int | None
    byte_size: int
    content_digest_blake2b_256: str
    mapping: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        expected_media = (
            "image/gif" if self.file_name.value.endswith(".gif") else "image/png"
        )
        if self.media_type != expected_media:
            raise ValueError("Artifact media_type disagrees with file suffix.")
        for name in ("width", "height", "frame_count", "byte_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"artifact.{name} must be a positive integer.")
        if expected_media == "image/gif":
            if (
                isinstance(self.duration_ms, bool)
                or not isinstance(self.duration_ms, int)
                or self.duration_ms <= 0
            ):
                raise ValueError("GIF artifacts require positive duration_ms.")
            if len(self.mapping) != self.frame_count:
                raise ValueError("GIF mapping length must equal frame_count.")
        elif self.duration_ms is not None or self.frame_count != 1:
            raise ValueError(
                "PNG artifacts require frame_count=1 and null duration_ms."
            )
        if (
            not isinstance(self.content_digest_blake2b_256, str)
            or len(self.content_digest_blake2b_256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.content_digest_blake2b_256
            )
        ):
            raise ValueError(
                "Artifact content digest must be 64 lowercase hex characters."
            )
        mapping = tuple(self.mapping)
        for value in mapping:
            _json_mapping(value, name="artifact.mapping")
        object.__setattr__(self, "mapping", mapping)

    def to_dict(self) -> dict[str, object]:
        """Return the exact manifest record."""
        return {
            "file_name": self.file_name.value,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_ms": self.duration_ms,
            "byte_size": self.byte_size,
            "content_digest_blake2b_256": self.content_digest_blake2b_256,
            "mapping": [dict(value) for value in self.mapping],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one exact artifact record."""
        raw = _exact_mapping(
            value,
            name="artifact",
            keys={
                "file_name",
                "media_type",
                "width",
                "height",
                "frame_count",
                "duration_ms",
                "byte_size",
                "content_digest_blake2b_256",
                "mapping",
            },
        )
        return cls(
            file_name=PublicationArtifactName(
                _text(raw["file_name"], name="file_name")
            ),
            media_type=_text(raw["media_type"], name="media_type"),
            width=_positive_integer(raw["width"], name="width"),
            height=_positive_integer(raw["height"], name="height"),
            frame_count=_positive_integer(raw["frame_count"], name="frame_count"),
            duration_ms=(
                None
                if raw["duration_ms"] is None
                else _positive_integer(raw["duration_ms"], name="duration_ms")
            ),
            byte_size=_positive_integer(raw["byte_size"], name="byte_size"),
            content_digest_blake2b_256=_text(
                raw["content_digest_blake2b_256"], name="content_digest_blake2b_256"
            ),
            mapping=tuple(
                _json_mapping(item, name="artifact.mapping")
                for item in _sequence(raw["mapping"], name="mapping")
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicationManifest:
    """Separately versioned exact semantic provenance for a complete bundle."""

    scene_id: str
    resolved_config: Mapping[str, object]
    source_owners: Mapping[str, object]
    artifacts: tuple[PublicationArtifactRecord, ...]
    coordinate_contract: str
    diagnostic_versions: Mapping[str, object]
    metrics: Mapping[str, object]
    asset_policy: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.scene_id, name="manifest.scene_id")
        if self.coordinate_contract != PUBLICATION_COORDINATE_CONTRACT:
            raise ValueError("Manifest coordinate contract is unsupported.")
        for field_value, name in (
            (self.resolved_config, "resolved_config"),
            (self.source_owners, "source_owners"),
            (self.diagnostic_versions, "diagnostic_versions"),
            (self.metrics, "metrics"),
            (self.asset_policy, "asset_policy"),
        ):
            _json_mapping(field_value, name=name)
        artifacts = tuple(self.artifacts)
        if (
            tuple(item.file_name for item in artifacts)
            != REQUIRED_PUBLICATION_ARTIFACTS
        ):
            raise ValueError(
                "Manifest artifacts do not match the complete fixed inventory."
            )
        expected_owners = {"court", "blcs", "plcs", "alignment", "reconstruction"}
        if set(self.source_owners) != expected_owners:
            raise ValueError("Manifest source owner inventory is missing or foreign.")
        for owner_name, owner_value in self.source_owners.items():
            owner = _json_mapping(owner_value, name=f"source_owners.{owner_name}")
            if owner.get("scene_id") != self.scene_id:
                raise ValueError("Every source owner must bind the manifest scene_id.")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, object]:
        """Return exact manifest JSON."""
        return {
            "schema": PUBLICATION_MANIFEST_SCHEMA,
            "bundle_schema": PUBLICATION_BUNDLE_SCHEMA,
            "request_schema": PUBLICATION_REQUEST_SCHEMA,
            "scene_id": self.scene_id,
            "resolved_config": dict(self.resolved_config),
            "source_owners": dict(self.source_owners),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "coordinate_contract": self.coordinate_contract,
            "diagnostic_versions": dict(self.diagnostic_versions),
            "metrics": dict(self.metrics),
            "asset_policy": dict(self.asset_policy),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse an exact manifest without aliases or inferred values."""
        raw = _exact_mapping(
            value,
            name="manifest",
            keys={
                "schema",
                "bundle_schema",
                "request_schema",
                "scene_id",
                "resolved_config",
                "source_owners",
                "artifacts",
                "coordinate_contract",
                "diagnostic_versions",
                "metrics",
                "asset_policy",
            },
        )
        if raw["schema"] != PUBLICATION_MANIFEST_SCHEMA:
            raise ValueError("Unsupported publication manifest schema.")
        if raw["bundle_schema"] != PUBLICATION_BUNDLE_SCHEMA:
            raise ValueError("Unsupported publication bundle schema.")
        if raw["request_schema"] != PUBLICATION_REQUEST_SCHEMA:
            raise ValueError("Unsupported publication request schema.")
        return cls(
            scene_id=_text(raw["scene_id"], name="scene_id"),
            resolved_config=_json_mapping(
                raw["resolved_config"], name="resolved_config"
            ),
            source_owners=_json_mapping(raw["source_owners"], name="source_owners"),
            artifacts=tuple(
                PublicationArtifactRecord.from_dict(item)
                for item in _sequence(raw["artifacts"], name="artifacts")
            ),
            coordinate_contract=_text(
                raw["coordinate_contract"], name="coordinate_contract"
            ),
            diagnostic_versions=_json_mapping(
                raw["diagnostic_versions"], name="diagnostic_versions"
            ),
            metrics=_json_mapping(raw["metrics"], name="metrics"),
            asset_policy=_json_mapping(raw["asset_policy"], name="asset_policy"),
        )


@dataclass(frozen=True, slots=True)
class PublicationBundleResult:
    """Published complete bundle and validated manifest."""

    bundle_path: Path
    manifest_path: Path
    manifest: PublicationManifest


def _absolute_path(value: Path, *, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path.")
    return value


def _identifier(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a portable identifier.")
    return text


def _identifiers(value: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty tuple.")
    result = tuple(_identifier(item, name=name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique.")
    return result


def _frame_indices(value: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
        or value != tuple(sorted(set(value)))
    ):
        raise ValueError(f"{name} must be non-empty, unique, and strictly increasing.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _exact_mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    result = _json_mapping(value, name=name)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _json_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    _json_value(value, name=name)
    return cast(Mapping[str, object], value)


def _json_value(value: object, *, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} contains a non-string key.")
        for key, item in value.items():
            _json_value(item, name=f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _json_value(item, name=f"{name}[{index}]")
        return
    raise TypeError(f"{name} contains a non-JSON value: {type(value).__name__}.")


__all__ = [
    "PUBLICATION_BUNDLE_SCHEMA",
    "PUBLICATION_COORDINATE_CONTRACT",
    "PUBLICATION_MANIFEST_SCHEMA",
    "PUBLICATION_REQUEST_SCHEMA",
    "PublicationArtifactName",
    "PublicationArtifactRecord",
    "PublicationBundleResult",
    "PublicationDrawingSettings",
    "PublicationManifest",
    "PublicationRequest",
    "REQUIRED_PUBLICATION_ARTIFACTS",
]
