"""Fail-closed normalization metadata for scene-directory datasets.

New datasets carry the same metadata object in the root ``meta.json`` and in
every scene ``meta.json``. Metadata-free datasets are recognized only as a
whole-artifact legacy state and only when the caller explicitly supplies the
``v1`` runtime contract. Versions are never inferred from shapes or values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    CourtCoordinateNormalizationVersion,
    resolve_court_coordinate_normalization,
)

COURT_COORDINATE_NORMALIZATION_METADATA_KEY: Final = (
    "court_coordinate_normalization"
)
COURT_COORDINATE_NORMALIZATION_METADATA_SCHEMA_VERSION: Final = 1
POSITION_UNIT_METRES: Final = "m"
VELOCITY_UNIT_METRES_PER_SECOND: Final = "m/s"

__all__ = [
    "COURT_COORDINATE_NORMALIZATION_METADATA_KEY",
    "COURT_COORDINATE_NORMALIZATION_METADATA_SCHEMA_VERSION",
    "CourtCoordinateArtifactContractError",
    "CourtCoordinateContractMismatchError",
    "CourtCoordinateNormalizationMetadata",
    "DatasetCourtCoordinateContract",
    "InvalidCourtCoordinateMetadataError",
    "MissingCourtCoordinateMetadataError",
    "MixedCourtCoordinateMetadataError",
    "POSITION_UNIT_METRES",
    "VELOCITY_UNIT_METRES_PER_SECOND",
    "extract_court_coordinate_normalization_metadata",
    "inject_court_coordinate_normalization_metadata",
    "validate_dataset_court_coordinate_contract",
    "validate_dataset_court_coordinate_contract_documents",
]

_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "version",
        "scale_xyz",
        "position_unit",
        "velocity_unit",
    }
)


class CourtCoordinateArtifactContractError(ValueError):
    """Base error for incompatible persisted normalization metadata."""


class MissingCourtCoordinateMetadataError(CourtCoordinateArtifactContractError):
    """Raised when a runtime is not allowed to consume metadata-free artifacts."""


class InvalidCourtCoordinateMetadataError(CourtCoordinateArtifactContractError):
    """Raised when a metadata object is incomplete, malformed, or unknown."""


class MixedCourtCoordinateMetadataError(CourtCoordinateArtifactContractError):
    """Raised when only part of a dataset carries normalization metadata."""


class CourtCoordinateContractMismatchError(CourtCoordinateArtifactContractError):
    """Raised when persisted and runtime normalization contracts differ."""


@dataclass(frozen=True, slots=True)
class CourtCoordinateNormalizationMetadata:
    """Immutable JSON schema shared by dataset and checkpoint artifacts."""

    schema_version: int
    version: CourtCoordinateNormalizationVersion
    scale_xyz: tuple[float, float, float]
    position_unit: Literal["m"]
    velocity_unit: Literal["m/s"]

    @property
    def contract(self) -> CourtCoordinateNormalization:
        """Return the resolver-owned mathematical contract."""
        return resolve_court_coordinate_normalization(self.version)

    @classmethod
    def from_contract(
        cls,
        contract: CourtCoordinateNormalization,
    ) -> CourtCoordinateNormalizationMetadata:
        """Build canonical artifact metadata for a resolved contract."""
        canonical = resolve_court_coordinate_normalization(contract.version)
        if contract != canonical:
            raise CourtCoordinateContractMismatchError(
                f"Runtime {contract.version!r} scale {contract.scale_xyz!r} does "
                f"not match the resolver scale {canonical.scale_xyz!r}."
            )
        return cls(
            schema_version=COURT_COORDINATE_NORMALIZATION_METADATA_SCHEMA_VERSION,
            version=canonical.version,
            scale_xyz=canonical.scale_xyz,
            position_unit=POSITION_UNIT_METRES,
            velocity_unit=VELOCITY_UNIT_METRES_PER_SECOND,
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        location: str,
    ) -> CourtCoordinateNormalizationMetadata:
        """Parse an exact metadata mapping and verify it against the resolver."""
        if not isinstance(value, Mapping):
            raise InvalidCourtCoordinateMetadataError(
                f"{location}: expected normalization metadata mapping, got "
                f"{type(value).__name__}."
            )
        non_string_keys = tuple(key for key in value if not isinstance(key, str))
        if non_string_keys:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}: metadata keys must be strings; got "
                f"{non_string_keys!r}."
            )
        keys = set(cast("Mapping[str, object]", value))
        missing = sorted(_METADATA_KEYS - keys)
        unknown = sorted(keys - _METADATA_KEYS)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing!r}")
            if unknown:
                details.append(f"unknown={unknown!r}")
            raise InvalidCourtCoordinateMetadataError(
                f"{location}: invalid normalization metadata fields "
                f"({', '.join(details)})."
            )
        mapping = cast("Mapping[str, object]", value)
        schema_version = mapping["schema_version"]
        if type(schema_version) is not int:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.schema_version: expected int, got "
                f"{type(schema_version).__name__}."
            )
        if schema_version != COURT_COORDINATE_NORMALIZATION_METADATA_SCHEMA_VERSION:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.schema_version: unsupported value "
                f"{schema_version!r}; expected "
                f"{COURT_COORDINATE_NORMALIZATION_METADATA_SCHEMA_VERSION}."
            )

        version = mapping["version"]
        if type(version) is not str:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.version: expected str, got {type(version).__name__}."
            )
        try:
            contract = resolve_court_coordinate_normalization(version)
        except ValueError as error:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.version: {error}"
            ) from error

        raw_scale = mapping["scale_xyz"]
        if (
            not isinstance(raw_scale, Sequence)
            or isinstance(raw_scale, (str, bytes, bytearray))
            or len(raw_scale) != 3
            or any(type(item) not in (int, float) for item in raw_scale)
        ):
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.scale_xyz: expected exactly three numbers; "
                f"got {raw_scale!r}."
            )
        scale_xyz = tuple(float(cast("int | float", item)) for item in raw_scale)
        if scale_xyz != contract.scale_xyz:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.scale_xyz: {scale_xyz!r} does not match "
                f"{contract.version!r} resolver scale {contract.scale_xyz!r}."
            )

        position_unit = mapping["position_unit"]
        velocity_unit = mapping["velocity_unit"]
        if position_unit != POSITION_UNIT_METRES:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.position_unit: expected 'm', got {position_unit!r}."
            )
        if velocity_unit != VELOCITY_UNIT_METRES_PER_SECOND:
            raise InvalidCourtCoordinateMetadataError(
                f"{location}.velocity_unit: expected 'm/s', got "
                f"{velocity_unit!r}."
            )
        return cls.from_contract(contract)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "scale_xyz": list(self.scale_xyz),
            "position_unit": self.position_unit,
            "velocity_unit": self.velocity_unit,
        }


@dataclass(frozen=True, slots=True)
class DatasetCourtCoordinateContract:
    """Successful dataset compatibility result."""

    contract: CourtCoordinateNormalization
    metadata: CourtCoordinateNormalizationMetadata | None
    legacy_metadata_free: bool
    scene_count: int


def extract_court_coordinate_normalization_metadata(
    document: Mapping[str, object],
    *,
    location: str,
) -> CourtCoordinateNormalizationMetadata | None:
    """Extract metadata, returning ``None`` only when the key is absent."""
    if COURT_COORDINATE_NORMALIZATION_METADATA_KEY not in document:
        return None
    return CourtCoordinateNormalizationMetadata.from_mapping(
        document[COURT_COORDINATE_NORMALIZATION_METADATA_KEY],
        location=f"{location}.{COURT_COORDINATE_NORMALIZATION_METADATA_KEY}",
    )


def inject_court_coordinate_normalization_metadata(
    document: Mapping[str, object],
    contract: CourtCoordinateNormalization,
    *,
    location: str,
) -> dict[str, object]:
    """Return a copy containing canonical metadata, rejecting a conflict."""
    result = dict(document)
    expected = CourtCoordinateNormalizationMetadata.from_contract(contract)
    existing = extract_court_coordinate_normalization_metadata(
        result,
        location=location,
    )
    if existing is not None and existing != expected:
        raise CourtCoordinateContractMismatchError(
            f"{location}: existing normalization metadata {existing.to_dict()!r} "
            f"does not match runtime {expected.to_dict()!r}."
        )
    result[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = expected.to_dict()
    return result


def validate_dataset_court_coordinate_contract_documents(
    *,
    root_metadata: Mapping[str, object],
    scene_metadata: Mapping[str, Mapping[str, object]],
    runtime_contract: CourtCoordinateNormalization,
    dataset_location: str = "dataset",
) -> DatasetCourtCoordinateContract:
    """Validate root and scene metadata before any scene arrays are used."""
    expected = CourtCoordinateNormalizationMetadata.from_contract(runtime_contract)
    entries: list[tuple[str, CourtCoordinateNormalizationMetadata | None]] = [
        (
            f"{dataset_location}/meta.json",
            extract_court_coordinate_normalization_metadata(
                root_metadata,
                location=f"{dataset_location}/meta.json",
            ),
        )
    ]
    entries.extend(
        (
            f"{dataset_location}/scenes/{name}/meta.json",
            extract_court_coordinate_normalization_metadata(
                metadata,
                location=f"{dataset_location}/scenes/{name}/meta.json",
            ),
        )
        for name, metadata in scene_metadata.items()
    )

    missing = [location for location, metadata in entries if metadata is None]
    present = [
        (location, metadata)
        for location, metadata in entries
        if metadata is not None
    ]
    if not present:
        if runtime_contract.version != "v1":
            raise MissingCourtCoordinateMetadataError(
                f"{dataset_location}: normalization metadata is absent at the root "
                "and all scenes. Metadata-free artifacts are legacy v1 only, but "
                f"runtime selected {runtime_contract.version!r}."
            )
        return DatasetCourtCoordinateContract(
            contract=runtime_contract,
            metadata=None,
            legacy_metadata_free=True,
            scene_count=len(scene_metadata),
        )
    if missing:
        raise MixedCourtCoordinateMetadataError(
            f"{dataset_location}: root/scene normalization metadata is mixed; "
            f"missing at {missing!r}."
        )

    for location, metadata in present:
        if metadata != expected:
            assert metadata is not None
            raise CourtCoordinateContractMismatchError(
                f"{location}: artifact normalization {metadata.to_dict()!r} does "
                f"not match runtime {expected.to_dict()!r}."
            )
    return DatasetCourtCoordinateContract(
        contract=runtime_contract,
        metadata=expected,
        legacy_metadata_free=False,
        scene_count=len(scene_metadata),
    )


def _load_json_mapping(
    path: Path,
    *,
    allow_absent_file: bool = False,
) -> dict[str, object]:
    try:
        value: Any = json.loads(path.read_text())
    except FileNotFoundError as error:
        if allow_absent_file:
            return {}
        raise MissingCourtCoordinateMetadataError(
            f"Required artifact metadata file does not exist: {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise InvalidCourtCoordinateMetadataError(
            f"{path}: invalid JSON metadata: {error}."
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InvalidCourtCoordinateMetadataError(
            f"{path}: expected a JSON object with string keys."
        )
    return cast("dict[str, object]", value)


def validate_dataset_court_coordinate_contract(
    dataset_root: str | Path,
    runtime_contract: CourtCoordinateNormalization,
    *,
    scene_paths: Sequence[str | Path] | None = None,
) -> DatasetCourtCoordinateContract:
    """Load and validate a dataset root plus every supplied scene header.

    When ``scene_paths`` is omitted, all directories immediately below
    ``<dataset_root>/scenes`` are checked. Callers using a split should pass all
    of that split's paths so validation happens before payload indexing.
    """
    root = Path(dataset_root)
    # Legacy datasets may predate the root header altogether. Treat that
    # absence only as an absent root contract entry so the whole-artifact
    # validator below can still reject v2, partial, and mixed metadata states.
    root_metadata = _load_json_mapping(
        root / "meta.json",
        allow_absent_file=True,
    )
    if scene_paths is None:
        scenes_dir = root / "scenes"
        if not scenes_dir.is_dir():
            raise MissingCourtCoordinateMetadataError(
                f"Dataset scenes directory does not exist: {scenes_dir}."
            )
        paths = sorted(path for path in scenes_dir.iterdir() if path.is_dir())
    else:
        paths = [Path(path) for path in scene_paths]

    by_name: dict[str, Mapping[str, object]] = {}
    for path in paths:
        if path.name in by_name:
            raise InvalidCourtCoordinateMetadataError(
                f"Duplicate scene name in contract validation: {path.name!r}."
            )
        by_name[path.name] = _load_json_mapping(path / "meta.json")
    return validate_dataset_court_coordinate_contract_documents(
        root_metadata=root_metadata,
        scene_metadata=by_name,
        runtime_contract=runtime_contract,
        dataset_location=str(root),
    )
