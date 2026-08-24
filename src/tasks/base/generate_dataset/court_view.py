"""Typed CourtKP20 artifact and reference-frame contracts.

The projector remains a physical-CourtKP20 primitive.  This module owns the
task-aware boundary used by BLCS and PLCS: explicit semantic contract
selection, camera-local disk ordering, strict metadata, reference alignment,
and reversible physical/reference frame transforms.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self, TypeAlias, cast, overload

import numpy as np
import torch
from torch import Tensor

from src.utils.schema.court import (
    COURT_KP20_HALF_TURN_INDEX,
    NUM_COURT_KP,
)

CourtKeypointSelector: TypeAlias = Literal["physical_v1", "camera_view_v2"]
CourtKeypointContractId: TypeAlias = Literal[
    "physical_courtkp20_v1",
    "camera_view_courtkp20_rzpi_v1",
]
CourtTargetFrameId: TypeAlias = Literal[
    "physical_court_v1",
    "reference_camera_court_rzpi_v1",
]
CourtCoordinateFrame: TypeAlias = Literal["physical_court"]
CourtArray: TypeAlias = np.ndarray | Tensor
Matrix3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

PHYSICAL_V1_SELECTOR: Final[CourtKeypointSelector] = "physical_v1"
CAMERA_VIEW_V2_SELECTOR: Final[CourtKeypointSelector] = "camera_view_v2"
PHYSICAL_COURTKP20_CONTRACT_ID: Final[CourtKeypointContractId] = "physical_courtkp20_v1"
CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID: Final[CourtKeypointContractId] = (
    "camera_view_courtkp20_rzpi_v1"
)
PHYSICAL_COURT_TARGET_FRAME_ID: Final[CourtTargetFrameId] = "physical_court_v1"
REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID: Final[CourtTargetFrameId] = (
    "reference_camera_court_rzpi_v1"
)
PHYSICAL_COURT_COORDINATE_FRAME: Final[CourtCoordinateFrame] = "physical_court"

COURT_KEYPOINT_METADATA_KEY: Final = "court_keypoints"
COURT_VIEW_METADATA_KEY: Final = "court_keypoint_views"
COURT_KEYPOINT_METADATA_SCHEMA_VERSION: Final = 1
COURT_VIEW_METADATA_SCHEMA_VERSION: Final = 1
REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION: Final = 1
CAMERA_MID_PLANE_EPSILON_M: Final = 1e-6

IDENTITY_COURT_KP20_INDEX: Final[tuple[int, ...]] = tuple(range(NUM_COURT_KP))
IDENTITY_ROTATION_3D: Final[Matrix3] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
RZ_PI_ROTATION_3D: Final[Matrix3] = (
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)

_CONTRACT_METADATA_FIELDS = frozenset(
    {"schema_version", "contract_id", "target_frame_id", "num_keypoints"}
)
_ARTIFACT_METADATA_FIELDS = frozenset(
    {"schema_version", "dataset_schema_id", "coordinate_frame", "contract"}
)
_COURT_VIEW_FIELDS = frozenset(
    {
        "schema_version",
        "camera_id",
        "contract_id",
        "semantic_to_physical",
        "canonical_from_physical",
        "camera_center_court_m",
        "coordinate_frame",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "target_frame_id",
        "reference_camera_id",
        "reference_camera_local_index",
        "reference_from_physical",
        "physical_from_reference",
    }
)


class CourtKeypointContractError(ValueError):
    """Base error for CourtKP20 semantic and frame contract violations."""


class UnknownCourtKeypointContractError(CourtKeypointContractError):
    """Raised when a selector or persisted contract ID is unsupported."""


class CourtKeypointMappingError(CourtKeypointContractError):
    """Raised when a semantic-to-physical permutation is malformed."""


class CameraCourtViewError(CourtKeypointContractError):
    """Raised when a camera cannot define the selected CourtKP20 view."""


class CourtKeypointMetadataError(CourtKeypointContractError):
    """Base error for persisted CourtKP20 metadata."""


class MissingCourtKeypointMetadataError(CourtKeypointMetadataError):
    """Raised when metadata is required but absent."""


class MixedCourtKeypointMetadataError(CourtKeypointMetadataError):
    """Raised when only part of an artifact has CourtKP20 metadata."""


class InvalidCourtKeypointMetadataError(CourtKeypointMetadataError):
    """Raised when metadata has invalid fields or values."""


class CourtKeypointContractMismatchError(CourtKeypointMetadataError):
    """Raised when two explicit CourtKP20 contracts are not identical."""


class CourtReferenceFrameError(CourtKeypointContractError):
    """Raised for invalid reference identity, provenance, or transforms."""


class CourtTransformShapeError(CourtReferenceFrameError):
    """Raised when a transformed value has an invalid shape."""


@dataclass(frozen=True, slots=True)
class CourtKeypointContract:
    """One canonical public selector and its semantic/model-frame IDs."""

    selector: CourtKeypointSelector
    contract_id: CourtKeypointContractId
    target_frame_id: CourtTargetFrameId

    def __post_init__(self) -> None:
        expected = _contract_definition(self.selector)
        if (self.contract_id, self.target_frame_id) != expected:
            raise CourtKeypointContractError(
                f"Court keypoint selector {self.selector!r} requires contract/target "
                f"IDs {expected!r}; got "
                f"{(self.contract_id, self.target_frame_id)!r}."
            )

    @property
    def camera_view_semantics(self) -> bool:
        """Whether disk Court channels use camera-local semantics."""
        return self.selector == CAMERA_VIEW_V2_SELECTOR


def _contract_definition(
    selector: str,
) -> tuple[CourtKeypointContractId, CourtTargetFrameId]:
    if selector == PHYSICAL_V1_SELECTOR:
        return PHYSICAL_COURTKP20_CONTRACT_ID, PHYSICAL_COURT_TARGET_FRAME_ID
    if selector == CAMERA_VIEW_V2_SELECTOR:
        return (
            CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID,
            REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID,
        )
    raise UnknownCourtKeypointContractError(
        f"Unknown court keypoint selector: {selector!r}. Supported selectors are "
        f"{PHYSICAL_V1_SELECTOR!r} and {CAMERA_VIEW_V2_SELECTOR!r}."
    )


def resolve_court_keypoint_contract(selector: str) -> CourtKeypointContract:
    """Resolve a public selector without inspecting arrays or metadata."""
    contract_id, target_frame_id = _contract_definition(selector)
    return CourtKeypointContract(
        selector=cast("CourtKeypointSelector", selector),
        contract_id=contract_id,
        target_frame_id=target_frame_id,
    )


def resolve_court_keypoint_contract_id(contract_id: str) -> CourtKeypointContract:
    """Resolve an exact persisted semantic ID; unknown IDs fail closed."""
    if contract_id == PHYSICAL_COURTKP20_CONTRACT_ID:
        return resolve_court_keypoint_contract(PHYSICAL_V1_SELECTOR)
    if contract_id == CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID:
        return resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    raise UnknownCourtKeypointContractError(
        f"Unknown court keypoint contract ID: {contract_id!r}. Supported IDs are "
        f"{PHYSICAL_COURTKP20_CONTRACT_ID!r} and "
        f"{CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID!r}."
    )


def _require_exact_mapping_fields(
    value: object,
    expected_fields: frozenset[str],
    *,
    location: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidCourtKeypointMetadataError(
            f"{location}: expected metadata mapping, got {type(value).__name__}."
        )
    if any(not isinstance(key, str) for key in value):
        raise InvalidCourtKeypointMetadataError(
            f"{location}: metadata keys must all be strings."
        )
    mapping = cast("Mapping[str, object]", value)
    fields = set(mapping)
    missing = sorted(expected_fields - fields)
    unknown = sorted(fields - expected_fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise InvalidCourtKeypointMetadataError(
            f"{location}: invalid metadata fields ({', '.join(details)})."
        )
    return mapping


def _require_schema_version(value: object, expected: int, *, location: str) -> None:
    if type(value) is not int:
        raise InvalidCourtKeypointMetadataError(
            f"{location}: expected int, got {type(value).__name__}."
        )
    if value != expected:
        raise InvalidCourtKeypointMetadataError(
            f"{location}: unsupported value {value!r}; expected {expected}."
        )


def _require_dataset_schema_id(value: object, *, location: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        raise InvalidCourtKeypointMetadataError(
            f"{location}: expected a non-empty lowercase schema ID, got {value!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class CourtKeypointContractMetadata:
    """JSON contract record shared by datasets, runtime inputs, and checkpoints."""

    schema_version: int
    contract_id: CourtKeypointContractId
    target_frame_id: CourtTargetFrameId
    num_keypoints: int

    @property
    def contract(self) -> CourtKeypointContract:
        """Return the canonical resolved semantic contract."""
        return resolve_court_keypoint_contract_id(self.contract_id)

    @classmethod
    def from_contract(
        cls,
        contract: CourtKeypointContract,
    ) -> CourtKeypointContractMetadata:
        """Build canonical metadata from a validated runtime contract."""
        canonical = resolve_court_keypoint_contract(contract.selector)
        if contract != canonical:
            raise CourtKeypointContractMismatchError(
                f"Runtime court keypoint contract is not canonical: {contract!r}."
            )
        return cls(
            schema_version=COURT_KEYPOINT_METADATA_SCHEMA_VERSION,
            contract_id=canonical.contract_id,
            target_frame_id=canonical.target_frame_id,
            num_keypoints=NUM_COURT_KP,
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        location: str,
    ) -> CourtKeypointContractMetadata:
        """Parse an exact contract mapping without version inference."""
        mapping = _require_exact_mapping_fields(
            value,
            _CONTRACT_METADATA_FIELDS,
            location=location,
        )
        _require_schema_version(
            mapping["schema_version"],
            COURT_KEYPOINT_METADATA_SCHEMA_VERSION,
            location=f"{location}.schema_version",
        )
        contract_id = mapping["contract_id"]
        if type(contract_id) is not str:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.contract_id: expected str, got "
                f"{type(contract_id).__name__}."
            )
        try:
            contract = resolve_court_keypoint_contract_id(contract_id)
        except UnknownCourtKeypointContractError as error:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.contract_id: {error}"
            ) from error
        target_frame_id = mapping["target_frame_id"]
        if target_frame_id != contract.target_frame_id:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.target_frame_id: expected "
                f"{contract.target_frame_id!r} for {contract.contract_id!r}, got "
                f"{target_frame_id!r}."
            )
        num_keypoints = mapping["num_keypoints"]
        if type(num_keypoints) is not int or num_keypoints != NUM_COURT_KP:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.num_keypoints: expected {NUM_COURT_KP}, got "
                f"{num_keypoints!r}."
            )
        return cls.from_contract(contract)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable exact record."""
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "target_frame_id": self.target_frame_id,
            "num_keypoints": self.num_keypoints,
        }


@dataclass(frozen=True, slots=True)
class CourtKeypointArtifactMetadata:
    """Root/scene record coupling a task dataset schema to CourtKP semantics."""

    schema_version: int
    dataset_schema_id: str
    coordinate_frame: CourtCoordinateFrame
    contract_metadata: CourtKeypointContractMetadata

    @property
    def contract(self) -> CourtKeypointContract:
        """Return the recorded semantic contract."""
        return self.contract_metadata.contract

    @classmethod
    def from_contract(
        cls,
        contract: CourtKeypointContract,
        *,
        dataset_schema_id: str,
    ) -> CourtKeypointArtifactMetadata:
        """Build canonical root/scene metadata."""
        schema_id = _require_dataset_schema_id(
            dataset_schema_id,
            location="dataset_schema_id",
        )
        return cls(
            schema_version=COURT_KEYPOINT_METADATA_SCHEMA_VERSION,
            dataset_schema_id=schema_id,
            coordinate_frame=PHYSICAL_COURT_COORDINATE_FRAME,
            contract_metadata=CourtKeypointContractMetadata.from_contract(contract),
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        location: str,
    ) -> CourtKeypointArtifactMetadata:
        """Parse an exact root/scene artifact mapping."""
        mapping = _require_exact_mapping_fields(
            value,
            _ARTIFACT_METADATA_FIELDS,
            location=location,
        )
        _require_schema_version(
            mapping["schema_version"],
            COURT_KEYPOINT_METADATA_SCHEMA_VERSION,
            location=f"{location}.schema_version",
        )
        dataset_schema_id = _require_dataset_schema_id(
            mapping["dataset_schema_id"],
            location=f"{location}.dataset_schema_id",
        )
        coordinate_frame = mapping["coordinate_frame"]
        if coordinate_frame != PHYSICAL_COURT_COORDINATE_FRAME:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.coordinate_frame: expected "
                f"{PHYSICAL_COURT_COORDINATE_FRAME!r}, got {coordinate_frame!r}."
            )
        contract_metadata = CourtKeypointContractMetadata.from_mapping(
            mapping["contract"],
            location=f"{location}.contract",
        )
        return cls(
            schema_version=COURT_KEYPOINT_METADATA_SCHEMA_VERSION,
            dataset_schema_id=dataset_schema_id,
            coordinate_frame=PHYSICAL_COURT_COORDINATE_FRAME,
            contract_metadata=contract_metadata,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable exact record."""
        return {
            "schema_version": self.schema_version,
            "dataset_schema_id": self.dataset_schema_id,
            "coordinate_frame": self.coordinate_frame,
            "contract": self.contract_metadata.to_dict(),
        }


def validate_court_keypoint_mapping(
    value: object,
    *,
    location: str = "semantic_to_physical",
) -> tuple[int, ...]:
    """Validate an exact length-20 bijective involution."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise CourtKeypointMappingError(
                f"{location}: expected shape ({NUM_COURT_KP},), got {value.shape!r}."
            )
        raw: Sequence[object] = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = value
    else:
        raise CourtKeypointMappingError(
            f"{location}: expected a length-{NUM_COURT_KP} integer sequence, got "
            f"{type(value).__name__}."
        )
    if len(raw) != NUM_COURT_KP:
        raise CourtKeypointMappingError(
            f"{location}: expected {NUM_COURT_KP} entries, got {len(raw)}."
        )
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer))
        for item in raw
    ):
        raise CourtKeypointMappingError(
            f"{location}: every entry must be an integer; got {tuple(raw)!r}."
        )
    mapping = tuple(int(cast("int | np.integer[Any]", item)) for item in raw)
    if set(mapping) != set(range(NUM_COURT_KP)):
        raise CourtKeypointMappingError(
            f"{location}: expected a bijection over 0..{NUM_COURT_KP - 1}; got "
            f"{mapping!r}."
        )
    if tuple(mapping[index] for index in mapping) != IDENTITY_COURT_KP20_INDEX:
        raise CourtKeypointMappingError(
            f"{location}: mapping must be an involution; got {mapping!r}."
        )
    return mapping


def _parse_numeric_vector3(
    value: object, *, location: str
) -> tuple[float, float, float]:
    if isinstance(value, np.ndarray):
        if value.shape != (3,):
            raise CameraCourtViewError(
                f"{location}: expected shape (3,), got {value.shape!r}."
            )
        raw: Sequence[object] = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = value
    else:
        raise CameraCourtViewError(
            f"{location}: expected exactly three numbers, got {type(value).__name__}."
        )
    if len(raw) != 3 or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, float, np.integer, np.floating))
        for item in raw
    ):
        raise CameraCourtViewError(
            f"{location}: expected exactly three numbers, got {tuple(raw)!r}."
        )
    result = tuple(
        float(cast("int | float | np.integer[Any] | np.floating[Any]", item))
        for item in raw
    )
    if not np.isfinite(np.asarray(result, dtype=np.float64)).all():
        raise CameraCourtViewError(
            f"{location}: camera center must contain only finite metres; got "
            f"{result!r}."
        )
    return cast("tuple[float, float, float]", result)


def _parse_matrix3(value: object, *, location: str) -> Matrix3:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CameraCourtViewError(
            f"{location}: expected a numeric 3x3 matrix."
        ) from error
    if array.shape != (3, 3):
        raise CameraCourtViewError(
            f"{location}: expected shape (3, 3), got {array.shape!r}."
        )
    if not np.isfinite(array).all():
        raise CameraCourtViewError(
            f"{location}: matrix must contain only finite values."
        )
    return cast("Matrix3", tuple(tuple(float(item) for item in row) for row in array))


def classify_camera_court_side(
    camera_center_court_m: object,
    *,
    epsilon_m: float = CAMERA_MID_PLANE_EPSILON_M,
) -> Literal["negative_y", "positive_y"]:
    """Classify a finite physical camera center, rejecting the mid-plane."""
    if not np.isfinite(epsilon_m) or epsilon_m < 0.0:
        raise CameraCourtViewError(
            f"epsilon_m must be finite and non-negative; got {epsilon_m!r}."
        )
    center = _parse_numeric_vector3(
        camera_center_court_m,
        location="camera_center_court_m",
    )
    if abs(center[1]) <= epsilon_m:
        raise CameraCourtViewError(
            "camera_center_court_m lies on the rejected court mid-plane: "
            f"abs(C_y)={abs(center[1])!r} <= {epsilon_m!r} m."
        )
    return "negative_y" if center[1] < 0.0 else "positive_y"


def _expected_camera_view(
    contract: CourtKeypointContract,
    center: tuple[float, float, float],
) -> tuple[tuple[int, ...], Matrix3]:
    if not contract.camera_view_semantics:
        return IDENTITY_COURT_KP20_INDEX, IDENTITY_ROTATION_3D
    side = classify_camera_court_side(center)
    if side == "negative_y":
        return IDENTITY_COURT_KP20_INDEX, IDENTITY_ROTATION_3D
    return COURT_KP20_HALF_TURN_INDEX, RZ_PI_ROTATION_3D


@dataclass(frozen=True, slots=True)
class CourtViewRecord:
    """Immutable per-camera disk ordering and physical-to-canonical rotation."""

    schema_version: int
    camera_id: str
    contract_id: CourtKeypointContractId
    semantic_to_physical: tuple[int, ...]
    canonical_from_physical: Matrix3
    camera_center_court_m: tuple[float, float, float]
    coordinate_frame: CourtCoordinateFrame

    def __post_init__(self) -> None:
        if self.schema_version != COURT_VIEW_METADATA_SCHEMA_VERSION:
            raise CameraCourtViewError(
                f"Court view schema_version must be "
                f"{COURT_VIEW_METADATA_SCHEMA_VERSION}; got {self.schema_version!r}."
            )
        if type(self.camera_id) is not str or not self.camera_id.strip():
            raise CameraCourtViewError(
                f"camera_id must be a non-empty string; got {self.camera_id!r}."
            )
        if self.coordinate_frame != PHYSICAL_COURT_COORDINATE_FRAME:
            raise CameraCourtViewError(
                f"coordinate_frame must be {PHYSICAL_COURT_COORDINATE_FRAME!r}; "
                f"got {self.coordinate_frame!r}."
            )
        try:
            contract = resolve_court_keypoint_contract_id(self.contract_id)
        except UnknownCourtKeypointContractError as error:
            raise CameraCourtViewError(str(error)) from error
        mapping = validate_court_keypoint_mapping(self.semantic_to_physical)
        center = _parse_numeric_vector3(
            self.camera_center_court_m,
            location=f"camera {self.camera_id!r}.camera_center_court_m",
        )
        matrix = _parse_matrix3(
            self.canonical_from_physical,
            location=f"camera {self.camera_id!r}.canonical_from_physical",
        )
        expected_mapping, expected_matrix = _expected_camera_view(contract, center)
        if mapping != expected_mapping:
            raise CameraCourtViewError(
                f"camera {self.camera_id!r}: semantic_to_physical {mapping!r} "
                f"does not match {contract.contract_id!r} at C_y={center[1]!r}; "
                f"expected {expected_mapping!r}."
            )
        if matrix != expected_matrix:
            raise CameraCourtViewError(
                f"camera {self.camera_id!r}: canonical_from_physical {matrix!r} "
                f"does not match {contract.contract_id!r} at C_y={center[1]!r}; "
                f"expected {expected_matrix!r}."
            )

    @property
    def contract(self) -> CourtKeypointContract:
        """Return the resolved semantic contract."""
        return resolve_court_keypoint_contract_id(self.contract_id)

    @classmethod
    def from_mapping(cls, value: object, *, location: str) -> CourtViewRecord:
        """Parse and fully validate one exact camera metadata record."""
        mapping = _require_exact_mapping_fields(
            value,
            _COURT_VIEW_FIELDS,
            location=location,
        )
        _require_schema_version(
            mapping["schema_version"],
            COURT_VIEW_METADATA_SCHEMA_VERSION,
            location=f"{location}.schema_version",
        )
        camera_id = mapping["camera_id"]
        contract_id = mapping["contract_id"]
        coordinate_frame = mapping["coordinate_frame"]
        if type(camera_id) is not str:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.camera_id: expected str, got {type(camera_id).__name__}."
            )
        if type(contract_id) is not str:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.contract_id: expected str, got "
                f"{type(contract_id).__name__}."
            )
        if coordinate_frame != PHYSICAL_COURT_COORDINATE_FRAME:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.coordinate_frame: expected "
                f"{PHYSICAL_COURT_COORDINATE_FRAME!r}, got {coordinate_frame!r}."
            )
        try:
            parsed_mapping = validate_court_keypoint_mapping(
                mapping["semantic_to_physical"],
                location=f"{location}.semantic_to_physical",
            )
            matrix = _parse_matrix3(
                mapping["canonical_from_physical"],
                location=f"{location}.canonical_from_physical",
            )
            center = _parse_numeric_vector3(
                mapping["camera_center_court_m"],
                location=f"{location}.camera_center_court_m",
            )
            return cls(
                schema_version=COURT_VIEW_METADATA_SCHEMA_VERSION,
                camera_id=camera_id,
                contract_id=cast("CourtKeypointContractId", contract_id),
                semantic_to_physical=parsed_mapping,
                canonical_from_physical=matrix,
                camera_center_court_m=center,
                coordinate_frame=PHYSICAL_COURT_COORDINATE_FRAME,
            )
        except (CourtKeypointMappingError, CameraCourtViewError) as error:
            raise InvalidCourtKeypointMetadataError(f"{location}: {error}") from error

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable exact camera record."""
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "contract_id": self.contract_id,
            "semantic_to_physical": list(self.semantic_to_physical),
            "canonical_from_physical": [
                list(row) for row in self.canonical_from_physical
            ],
            "camera_center_court_m": list(self.camera_center_court_m),
            "coordinate_frame": self.coordinate_frame,
        }


def build_court_view_record(
    *,
    camera_id: str,
    camera_center_court_m: object,
    contract: CourtKeypointContract,
) -> CourtViewRecord:
    """Build one canonical physical-v1 or camera-view-v2 camera record."""
    canonical_contract = resolve_court_keypoint_contract(contract.selector)
    if contract != canonical_contract:
        raise CourtKeypointContractMismatchError(
            f"Runtime court keypoint contract is not canonical: {contract!r}."
        )
    center = _parse_numeric_vector3(
        camera_center_court_m,
        location=f"camera {camera_id!r}.camera_center_court_m",
    )
    mapping, matrix = _expected_camera_view(contract, center)
    return CourtViewRecord(
        schema_version=COURT_VIEW_METADATA_SCHEMA_VERSION,
        camera_id=camera_id,
        contract_id=contract.contract_id,
        semantic_to_physical=mapping,
        canonical_from_physical=matrix,
        camera_center_court_m=center,
        coordinate_frame=PHYSICAL_COURT_COORDINATE_FRAME,
    )


def _validate_view_records(
    records: Sequence[CourtViewRecord],
    *,
    contract: CourtKeypointContract,
    location: str,
) -> tuple[CourtViewRecord, ...]:
    if not records:
        raise InvalidCourtKeypointMetadataError(
            f"{location}: at least one camera court-view record is required."
        )
    validated = tuple(records)
    camera_ids = tuple(record.camera_id for record in validated)
    if len(set(camera_ids)) != len(camera_ids):
        raise InvalidCourtKeypointMetadataError(
            f"{location}: camera_id values must be unique; got {camera_ids!r}."
        )
    mismatches = tuple(
        record.camera_id
        for record in validated
        if record.contract_id != contract.contract_id
    )
    if mismatches:
        raise CourtKeypointContractMismatchError(
            f"{location}: cameras {mismatches!r} do not use root/scene contract "
            f"{contract.contract_id!r}."
        )
    return validated


def extract_court_keypoint_artifact_metadata(
    document: Mapping[str, object],
    *,
    location: str,
) -> CourtKeypointArtifactMetadata | None:
    """Extract root/scene metadata, returning ``None`` only for an absent key."""
    if COURT_KEYPOINT_METADATA_KEY not in document:
        return None
    return CourtKeypointArtifactMetadata.from_mapping(
        document[COURT_KEYPOINT_METADATA_KEY],
        location=f"{location}.{COURT_KEYPOINT_METADATA_KEY}",
    )


def extract_court_keypoint_contract_metadata(
    document: Mapping[str, object],
    *,
    location: str,
) -> CourtKeypointContractMetadata | None:
    """Extract the shared model/runtime contract record from a document."""
    if COURT_KEYPOINT_METADATA_KEY not in document:
        return None
    return CourtKeypointContractMetadata.from_mapping(
        document[COURT_KEYPOINT_METADATA_KEY],
        location=f"{location}.{COURT_KEYPOINT_METADATA_KEY}",
    )


def extract_court_view_records(
    document: Mapping[str, object],
    *,
    contract: CourtKeypointContract,
    location: str,
) -> tuple[CourtViewRecord, ...] | None:
    """Extract a scene's ordered per-camera records without filling omissions."""
    if COURT_VIEW_METADATA_KEY not in document:
        return None
    raw = document[COURT_VIEW_METADATA_KEY]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise InvalidCourtKeypointMetadataError(
            f"{location}.{COURT_VIEW_METADATA_KEY}: expected a camera record list."
        )
    records = tuple(
        CourtViewRecord.from_mapping(
            value,
            location=f"{location}.{COURT_VIEW_METADATA_KEY}[{index}]",
        )
        for index, value in enumerate(raw)
    )
    return _validate_view_records(
        records,
        contract=contract,
        location=f"{location}.{COURT_VIEW_METADATA_KEY}",
    )


def inject_court_keypoint_artifact_metadata(
    document: Mapping[str, object],
    metadata: CourtKeypointArtifactMetadata,
    *,
    location: str,
) -> dict[str, object]:
    """Copy a root document and inject canonical metadata without replacement."""
    result = dict(document)
    existing = extract_court_keypoint_artifact_metadata(result, location=location)
    if existing is not None and existing != metadata:
        raise CourtKeypointContractMismatchError(
            f"{location}: refusing to replace court keypoint metadata "
            f"{existing.to_dict()!r} with {metadata.to_dict()!r}."
        )
    if COURT_VIEW_METADATA_KEY in result:
        raise InvalidCourtKeypointMetadataError(
            f"{location}: root metadata must not contain per-camera "
            f"{COURT_VIEW_METADATA_KEY!r}."
        )
    result[COURT_KEYPOINT_METADATA_KEY] = metadata.to_dict()
    return result


def inject_scene_court_keypoint_metadata(
    document: Mapping[str, object],
    metadata: CourtKeypointArtifactMetadata,
    court_views: Sequence[CourtViewRecord],
    *,
    location: str,
) -> dict[str, object]:
    """Copy a scene document and inject exact scene/camera metadata."""
    result = dict(document)
    existing = extract_court_keypoint_artifact_metadata(result, location=location)
    if existing is not None and existing != metadata:
        raise CourtKeypointContractMismatchError(
            f"{location}: refusing to replace court keypoint metadata "
            f"{existing.to_dict()!r} with {metadata.to_dict()!r}."
        )
    records = _validate_view_records(
        court_views,
        contract=metadata.contract,
        location=f"{location}.{COURT_VIEW_METADATA_KEY}",
    )
    if COURT_VIEW_METADATA_KEY in result:
        parsed = extract_court_view_records(
            result,
            contract=metadata.contract,
            location=location,
        )
        if parsed != records:
            raise CourtKeypointContractMismatchError(
                f"{location}: refusing to replace existing per-camera court "
                "keypoint records."
            )
    result[COURT_KEYPOINT_METADATA_KEY] = metadata.to_dict()
    result[COURT_VIEW_METADATA_KEY] = [record.to_dict() for record in records]
    return result


@dataclass(frozen=True, slots=True)
class SceneCourtViewRecords:
    """Validated camera metadata for one stable scene identity."""

    scene_id: str
    court_views: tuple[CourtViewRecord, ...]


@dataclass(frozen=True, slots=True)
class DatasetCourtKeypointContract:
    """Successful root/scene/camera compatibility result."""

    contract: CourtKeypointContract
    metadata: CourtKeypointArtifactMetadata | None
    legacy_metadata_free: bool
    scenes: tuple[SceneCourtViewRecords, ...]


def validate_dataset_court_keypoint_contract_documents(
    *,
    root_metadata: Mapping[str, object],
    scene_metadata: Mapping[str, Mapping[str, object]],
    runtime_contract: CourtKeypointContract,
    expected_dataset_schema_id: str,
    dataset_location: str = "dataset",
) -> DatasetCourtKeypointContract:
    """Validate all root/scene/camera metadata before payload arrays are used."""
    canonical_contract = resolve_court_keypoint_contract(runtime_contract.selector)
    if runtime_contract != canonical_contract:
        raise CourtKeypointContractMismatchError(
            f"{dataset_location}: runtime contract is not canonical: "
            f"{runtime_contract!r}."
        )
    expected = CourtKeypointArtifactMetadata.from_contract(
        runtime_contract,
        dataset_schema_id=expected_dataset_schema_id,
    )
    root_location = f"{dataset_location}/meta.json"
    root_entry = extract_court_keypoint_artifact_metadata(
        root_metadata,
        location=root_location,
    )
    if COURT_VIEW_METADATA_KEY in root_metadata:
        raise InvalidCourtKeypointMetadataError(
            f"{root_location}: root metadata must not contain per-camera "
            f"{COURT_VIEW_METADATA_KEY!r}."
        )

    scene_entries: list[
        tuple[
            str,
            str,
            CourtKeypointArtifactMetadata | None,
            tuple[CourtViewRecord, ...] | None,
        ]
    ] = []
    for scene_id, document in scene_metadata.items():
        location = f"{dataset_location}/scenes/{scene_id}/meta.json"
        artifact = extract_court_keypoint_artifact_metadata(
            document,
            location=location,
        )
        views = (
            extract_court_view_records(
                document,
                contract=artifact.contract,
                location=location,
            )
            if artifact is not None
            else None
        )
        if artifact is None and COURT_VIEW_METADATA_KEY in document:
            raise MixedCourtKeypointMetadataError(
                f"{location}: per-camera metadata is present without scene "
                "court keypoint metadata."
            )
        scene_entries.append((scene_id, location, artifact, views))

    artifact_entries = [(root_location, root_entry)] + [
        (location, artifact) for _, location, artifact, _ in scene_entries
    ]
    present = [(location, item) for location, item in artifact_entries if item]
    missing = [location for location, item in artifact_entries if item is None]
    view_present = any(
        COURT_VIEW_METADATA_KEY in document for document in scene_metadata.values()
    )
    if not present:
        if view_present:
            raise MixedCourtKeypointMetadataError(
                f"{dataset_location}: camera metadata exists without root/scene "
                "court keypoint metadata."
            )
        if runtime_contract.selector != PHYSICAL_V1_SELECTOR:
            raise MissingCourtKeypointMetadataError(
                f"{dataset_location}: court keypoint metadata is absent. "
                "Metadata-free artifacts are accepted only by an explicitly "
                f"selected {PHYSICAL_V1_SELECTOR!r} runtime; got "
                f"{runtime_contract.selector!r}."
            )
        return DatasetCourtKeypointContract(
            contract=runtime_contract,
            metadata=None,
            legacy_metadata_free=True,
            scenes=tuple(
                SceneCourtViewRecords(scene_id=scene_id, court_views=())
                for scene_id in scene_metadata
            ),
        )
    if missing:
        raise MixedCourtKeypointMetadataError(
            f"{dataset_location}: root/scene court keypoint metadata is mixed; "
            f"missing at {missing!r}."
        )

    for location, item in present:
        if item != expected:
            assert item is not None
            raise CourtKeypointContractMismatchError(
                f"{location}: artifact court keypoint metadata "
                f"{item.to_dict()!r} does not exactly match runtime "
                f"{expected.to_dict()!r}."
            )

    scenes: list[SceneCourtViewRecords] = []
    for scene_id, location, _, views in scene_entries:
        if views is None:
            raise MixedCourtKeypointMetadataError(
                f"{location}: scene contract exists but "
                f"{COURT_VIEW_METADATA_KEY!r} is missing."
            )
        scenes.append(SceneCourtViewRecords(scene_id=scene_id, court_views=views))
    return DatasetCourtKeypointContract(
        contract=runtime_contract,
        metadata=expected,
        legacy_metadata_free=False,
        scenes=tuple(scenes),
    )


def _load_json_mapping(
    path: Path,
    *,
    allow_absent_file: bool = False,
) -> dict[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        if allow_absent_file:
            return {}
        raise MissingCourtKeypointMetadataError(
            f"Required artifact metadata file does not exist: {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise InvalidCourtKeypointMetadataError(
            f"{path}: invalid JSON metadata: {error}."
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InvalidCourtKeypointMetadataError(
            f"{path}: expected a JSON object with string keys."
        )
    return cast("dict[str, object]", value)


def validate_dataset_court_keypoint_contract(
    dataset_root: str | Path,
    runtime_contract: CourtKeypointContract,
    *,
    expected_dataset_schema_id: str,
    scene_paths: Sequence[str | Path] | None = None,
) -> DatasetCourtKeypointContract:
    """Load and validate a dataset root plus all requested scene headers."""
    root = Path(dataset_root)
    root_metadata = _load_json_mapping(
        root / "meta.json",
        allow_absent_file=True,
    )
    if scene_paths is None:
        scenes_dir = root / "scenes"
        if not scenes_dir.is_dir():
            raise MissingCourtKeypointMetadataError(
                f"Dataset scenes directory does not exist: {scenes_dir}."
            )
        paths = sorted(path for path in scenes_dir.iterdir() if path.is_dir())
    else:
        paths = [Path(path) for path in scene_paths]
    by_name: dict[str, Mapping[str, object]] = {}
    for path in paths:
        if path.name in by_name:
            raise InvalidCourtKeypointMetadataError(
                f"Duplicate scene name in contract validation: {path.name!r}."
            )
        by_name[path.name] = _load_json_mapping(path / "meta.json")
    return validate_dataset_court_keypoint_contract_documents(
        root_metadata=root_metadata,
        scene_metadata=by_name,
        runtime_contract=runtime_contract,
        expected_dataset_schema_id=expected_dataset_schema_id,
        dataset_location=str(root),
    )


def _normalize_axis(axis: int, ndim: int, *, location: str) -> int:
    normalized = axis if axis >= 0 else ndim + axis
    if normalized < 0 or normalized >= ndim:
        raise CourtTransformShapeError(
            f"{location}: axis {axis} is invalid for rank {ndim}."
        )
    return normalized


@overload
def reorder_court_keypoints(
    value: np.ndarray,
    semantic_to_physical: object,
    *,
    keypoint_axis: int,
) -> np.ndarray: ...


@overload
def reorder_court_keypoints(
    value: Tensor,
    semantic_to_physical: object,
    *,
    keypoint_axis: int,
) -> Tensor: ...


def reorder_court_keypoints(
    value: CourtArray,
    semantic_to_physical: object,
    *,
    keypoint_axis: int,
) -> CourtArray:
    """Index a physical CourtKP20 array into one validated semantic ordering."""
    if not isinstance(value, (np.ndarray, Tensor)):
        raise TypeError(
            "Court keypoints must be a numpy.ndarray or torch.Tensor; got "
            f"{type(value).__name__}."
        )
    mapping = validate_court_keypoint_mapping(semantic_to_physical)
    axis = _normalize_axis(keypoint_axis, value.ndim, location="court keypoints")
    if value.shape[axis] != NUM_COURT_KP:
        raise CourtTransformShapeError(
            f"court keypoints: axis {keypoint_axis} must have length "
            f"{NUM_COURT_KP}; got shape {tuple(value.shape)!r}."
        )
    if isinstance(value, Tensor):
        index = torch.tensor(mapping, dtype=torch.long, device=value.device)
        return torch.index_select(value, axis, index)
    return np.take(value, np.asarray(mapping, dtype=np.intp), axis=axis)


@overload
def apply_court_view_record(
    value: np.ndarray,
    court_view: CourtViewRecord,
    *,
    keypoint_axis: int,
) -> np.ndarray: ...


@overload
def apply_court_view_record(
    value: Tensor,
    court_view: CourtViewRecord,
    *,
    keypoint_axis: int,
) -> Tensor: ...


def apply_court_view_record(
    value: CourtArray,
    court_view: CourtViewRecord,
    *,
    keypoint_axis: int,
) -> CourtArray:
    """Convert physical projection/visibility slots to disk semantic slots."""
    return reorder_court_keypoints(
        value,
        court_view.semantic_to_physical,
        keypoint_axis=keypoint_axis,
    )


def reference_court_keypoint_indices(
    source_view: CourtViewRecord,
    reference_view: CourtViewRecord,
) -> tuple[int, ...]:
    """Return ``H_source^-1 o H_reference`` indices for disk arrays."""
    if source_view.contract_id != reference_view.contract_id:
        raise CourtKeypointContractMismatchError(
            f"Cannot align camera {source_view.camera_id!r} contract "
            f"{source_view.contract_id!r} to camera {reference_view.camera_id!r} "
            f"contract {reference_view.contract_id!r}."
        )
    inverse_source = [0] * NUM_COURT_KP
    for semantic_index, physical_index in enumerate(source_view.semantic_to_physical):
        inverse_source[physical_index] = semantic_index
    remap = tuple(
        inverse_source[physical_index]
        for physical_index in reference_view.semantic_to_physical
    )
    return validate_court_keypoint_mapping(
        remap,
        location=(f"H_{source_view.camera_id}^-1 o H_{reference_view.camera_id}"),
    )


@overload
def align_court_keypoints_to_reference(
    value: np.ndarray,
    source_view: CourtViewRecord,
    reference_view: CourtViewRecord,
    *,
    keypoint_axis: int,
) -> np.ndarray: ...


@overload
def align_court_keypoints_to_reference(
    value: Tensor,
    source_view: CourtViewRecord,
    reference_view: CourtViewRecord,
    *,
    keypoint_axis: int,
) -> Tensor: ...


def align_court_keypoints_to_reference(
    value: CourtArray,
    source_view: CourtViewRecord,
    reference_view: CourtViewRecord,
    *,
    keypoint_axis: int,
) -> CourtArray:
    """Align one camera-local disk Court array to reference semantic order."""
    return reorder_court_keypoints(
        value,
        reference_court_keypoint_indices(source_view, reference_view),
        keypoint_axis=keypoint_axis,
    )


def resolve_reference_camera_local_index(
    selected_camera_ids: Sequence[str],
    reference_camera_id: str,
) -> int:
    """Resolve a stable reference identity after subset/reordering."""
    ids = tuple(selected_camera_ids)
    if not ids:
        raise CourtReferenceFrameError(
            "selected_camera_ids must contain at least one stable camera ID."
        )
    if any(type(camera_id) is not str or not camera_id.strip() for camera_id in ids):
        raise CourtReferenceFrameError(
            f"selected_camera_ids must be non-empty strings; got {ids!r}."
        )
    if len(set(ids)) != len(ids):
        raise CourtReferenceFrameError(
            f"selected_camera_ids must be unique; got {ids!r}."
        )
    if type(reference_camera_id) is not str or not reference_camera_id.strip():
        raise CourtReferenceFrameError(
            f"reference_camera_id must be a non-empty string; got "
            f"{reference_camera_id!r}."
        )
    try:
        return ids.index(reference_camera_id)
    except ValueError as error:
        raise CourtReferenceFrameError(
            f"reference camera {reference_camera_id!r} is not in selected cameras "
            f"{ids!r}."
        ) from error


def resolve_reference_court_view(
    selected_views: Sequence[CourtViewRecord],
    reference_camera_id: str,
) -> tuple[int, CourtViewRecord]:
    """Resolve one view while validating IDs independently of local ordering."""
    views = tuple(selected_views)
    index = resolve_reference_camera_local_index(
        tuple(view.camera_id for view in views),
        reference_camera_id,
    )
    contract_ids = {view.contract_id for view in views}
    if len(contract_ids) != 1:
        raise CourtKeypointContractMismatchError(
            f"Selected cameras have mixed court keypoint contracts: "
            f"{sorted(contract_ids)!r}."
        )
    return index, views[index]


def _validate_proper_rotation_pair(
    forward: Matrix3,
    inverse: Matrix3,
    *,
    location: str,
) -> None:
    forward_array = np.asarray(forward, dtype=np.float64)
    inverse_array = np.asarray(inverse, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(
        forward_array.T @ forward_array,
        identity,
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        np.linalg.det(forward_array),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise CourtReferenceFrameError(
            f"{location}.reference_from_physical must be a finite proper rotation."
        )
    if not np.allclose(
        inverse_array,
        forward_array.T,
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        forward_array @ inverse_array,
        identity,
        rtol=0.0,
        atol=1e-12,
    ):
        raise CourtReferenceFrameError(
            f"{location}: physical/reference matrices are not exact inverses."
        )


@dataclass(frozen=True, slots=True)
class CourtReferenceFrameProvenance:
    """Reversible model/prediction frame metadata for one selected sample."""

    schema_version: int
    contract_id: CourtKeypointContractId
    target_frame_id: CourtTargetFrameId
    reference_camera_id: str | None
    reference_camera_local_index: int | None
    reference_from_physical: Matrix3
    physical_from_reference: Matrix3

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION:
            raise CourtReferenceFrameError(
                "Reference provenance schema_version must be "
                f"{REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION}; got "
                f"{self.schema_version!r}."
            )
        contract = resolve_court_keypoint_contract_id(self.contract_id)
        if self.target_frame_id != contract.target_frame_id:
            raise CourtReferenceFrameError(
                f"target_frame_id must be {contract.target_frame_id!r} for "
                f"{contract.contract_id!r}; got {self.target_frame_id!r}."
            )
        forward = _parse_matrix3(
            self.reference_from_physical,
            location="reference_from_physical",
        )
        inverse = _parse_matrix3(
            self.physical_from_reference,
            location="physical_from_reference",
        )
        _validate_proper_rotation_pair(forward, inverse, location="provenance")
        if contract.selector == PHYSICAL_V1_SELECTOR:
            if (
                self.reference_camera_id is not None
                or self.reference_camera_local_index is not None
            ):
                raise CourtReferenceFrameError(
                    "physical-v1 provenance must not claim a reference camera."
                )
            if forward != IDENTITY_ROTATION_3D or inverse != IDENTITY_ROTATION_3D:
                raise CourtReferenceFrameError(
                    "physical-v1 provenance must use identity transforms."
                )
            return
        if (
            type(self.reference_camera_id) is not str
            or not self.reference_camera_id.strip()
        ):
            raise CourtReferenceFrameError(
                "camera-view-v2 provenance requires a non-empty reference_camera_id."
            )
        if (
            type(self.reference_camera_local_index) is not int
            or self.reference_camera_local_index < 0
        ):
            raise CourtReferenceFrameError(
                "camera-view-v2 provenance requires a non-negative integer "
                "reference_camera_local_index."
            )
        if forward not in (IDENTITY_ROTATION_3D, RZ_PI_ROTATION_3D):
            raise CourtReferenceFrameError(
                "camera-view-v2 reference_from_physical must be exactly identity "
                "or Rz(pi)."
            )

    @property
    def contract(self) -> CourtKeypointContract:
        """Return the resolved CourtKP20 semantic contract."""
        return resolve_court_keypoint_contract_id(self.contract_id)

    def to(self, *args: object, **kwargs: object) -> Self:
        """Preserve tensor-free immutable metadata during framework transfers.

        Device, dtype, and transfer arguments are intentionally ignored because
        this record contains no tensors and therefore has nothing to convert.
        """
        return self

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        location: str,
    ) -> CourtReferenceFrameProvenance:
        """Parse exact reversible prediction/sample provenance."""
        mapping = _require_exact_mapping_fields(
            value,
            _PROVENANCE_FIELDS,
            location=location,
        )
        _require_schema_version(
            mapping["schema_version"],
            REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION,
            location=f"{location}.schema_version",
        )
        contract_id = mapping["contract_id"]
        target_frame_id = mapping["target_frame_id"]
        reference_camera_id = mapping["reference_camera_id"]
        reference_camera_local_index = mapping["reference_camera_local_index"]
        if type(contract_id) is not str or type(target_frame_id) is not str:
            raise InvalidCourtKeypointMetadataError(
                f"{location}: contract_id and target_frame_id must be strings."
            )
        if reference_camera_id is not None and type(reference_camera_id) is not str:
            raise InvalidCourtKeypointMetadataError(
                f"{location}.reference_camera_id: expected str or null."
            )
        if reference_camera_local_index is not None and (
            type(reference_camera_local_index) is not int
        ):
            raise InvalidCourtKeypointMetadataError(
                f"{location}.reference_camera_local_index: expected int or null."
            )
        try:
            return cls(
                schema_version=REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION,
                contract_id=cast("CourtKeypointContractId", contract_id),
                target_frame_id=cast("CourtTargetFrameId", target_frame_id),
                reference_camera_id=reference_camera_id,
                reference_camera_local_index=reference_camera_local_index,
                reference_from_physical=_parse_matrix3(
                    mapping["reference_from_physical"],
                    location=f"{location}.reference_from_physical",
                ),
                physical_from_reference=_parse_matrix3(
                    mapping["physical_from_reference"],
                    location=f"{location}.physical_from_reference",
                ),
            )
        except (
            CameraCourtViewError,
            CourtReferenceFrameError,
            UnknownCourtKeypointContractError,
        ) as error:
            raise InvalidCourtKeypointMetadataError(f"{location}: {error}") from error

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable reversible provenance."""
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "target_frame_id": self.target_frame_id,
            "reference_camera_id": self.reference_camera_id,
            "reference_camera_local_index": self.reference_camera_local_index,
            "reference_from_physical": [
                list(row) for row in self.reference_from_physical
            ],
            "physical_from_reference": [
                list(row) for row in self.physical_from_reference
            ],
        }


def build_physical_court_provenance() -> CourtReferenceFrameProvenance:
    """Build explicit identity provenance for a physical-v1 model sample."""
    contract = resolve_court_keypoint_contract(PHYSICAL_V1_SELECTOR)
    return CourtReferenceFrameProvenance(
        schema_version=REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION,
        contract_id=contract.contract_id,
        target_frame_id=contract.target_frame_id,
        reference_camera_id=None,
        reference_camera_local_index=None,
        reference_from_physical=IDENTITY_ROTATION_3D,
        physical_from_reference=IDENTITY_ROTATION_3D,
    )


def build_reference_frame_provenance(
    selected_views: Sequence[CourtViewRecord],
    *,
    reference_camera_id: str,
) -> CourtReferenceFrameProvenance:
    """Build v2 provenance from one stable selected camera identity."""
    local_index, reference_view = resolve_reference_court_view(
        selected_views,
        reference_camera_id,
    )
    contract = reference_view.contract
    if contract.selector != CAMERA_VIEW_V2_SELECTOR:
        raise CourtReferenceFrameError(
            "Reference-camera provenance is valid only for the explicit "
            f"{CAMERA_VIEW_V2_SELECTOR!r} contract. Use "
            "build_physical_court_provenance() for physical v1."
        )
    forward = reference_view.canonical_from_physical
    inverse = cast(
        "Matrix3",
        tuple(tuple(row) for row in zip(*forward, strict=True)),
    )
    return CourtReferenceFrameProvenance(
        schema_version=REFERENCE_FRAME_PROVENANCE_SCHEMA_VERSION,
        contract_id=contract.contract_id,
        target_frame_id=contract.target_frame_id,
        reference_camera_id=reference_camera_id,
        reference_camera_local_index=local_index,
        reference_from_physical=forward,
        physical_from_reference=inverse,
    )


def validate_reference_frame_provenance(
    provenance: CourtReferenceFrameProvenance,
    selected_views: Sequence[CourtViewRecord],
) -> CourtViewRecord | None:
    """Cross-check serialized provenance against current selected view order."""
    contract = provenance.contract
    views = tuple(selected_views)
    if contract.selector == PHYSICAL_V1_SELECTOR:
        mismatched = tuple(
            view.camera_id for view in views if view.contract_id != contract.contract_id
        )
        if mismatched:
            raise CourtKeypointContractMismatchError(
                f"Physical provenance does not match selected cameras {mismatched!r}."
            )
        return None
    assert provenance.reference_camera_id is not None
    assert provenance.reference_camera_local_index is not None
    index, reference_view = resolve_reference_court_view(
        views,
        provenance.reference_camera_id,
    )
    if index != provenance.reference_camera_local_index:
        raise CourtReferenceFrameError(
            f"Reference camera {provenance.reference_camera_id!r} resolves to "
            f"local index {index}, but provenance records "
            f"{provenance.reference_camera_local_index}."
        )
    if reference_view.contract_id != provenance.contract_id:
        raise CourtKeypointContractMismatchError(
            f"Reference camera contract {reference_view.contract_id!r} does not "
            f"match provenance {provenance.contract_id!r}."
        )
    if reference_view.canonical_from_physical != provenance.reference_from_physical:
        raise CourtReferenceFrameError(
            f"Reference camera {reference_view.camera_id!r} transform does not "
            "match provenance."
        )
    return reference_view


def _require_court_array(
    value: CourtArray,
    *,
    trailing_shape: tuple[int, ...],
    quantity: str,
) -> None:
    if not isinstance(value, (np.ndarray, Tensor)):
        raise TypeError(
            f"{quantity} must be a numpy.ndarray or torch.Tensor; got "
            f"{type(value).__name__}."
        )
    if (
        value.ndim < len(trailing_shape)
        or tuple(value.shape[-len(trailing_shape) :]) != trailing_shape
    ):
        raise CourtTransformShapeError(
            f"{quantity} must have shape (..., {', '.join(map(str, trailing_shape))}); "
            f"got {tuple(value.shape)!r}."
        )
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise CourtReferenceFrameError(
                f"{quantity} must use a floating dtype; got {value.dtype}."
            )
        if not torch.isfinite(value).all().item():
            raise CourtReferenceFrameError(
                f"{quantity} must contain only finite values."
            )
    else:
        if not np.issubdtype(value.dtype, np.floating):
            raise CourtReferenceFrameError(
                f"{quantity} must use a floating dtype; got {value.dtype}."
            )
        if not np.isfinite(value).all():
            raise CourtReferenceFrameError(
                f"{quantity} must contain only finite values."
            )


@overload
def _rotate_last_dimension(
    value: np.ndarray,
    matrix: Matrix3,
    *,
    dimensions: int,
    quantity: str,
) -> np.ndarray: ...


@overload
def _rotate_last_dimension(
    value: Tensor,
    matrix: Matrix3,
    *,
    dimensions: int,
    quantity: str,
) -> Tensor: ...


def _rotate_last_dimension(
    value: CourtArray,
    matrix: Matrix3,
    *,
    dimensions: int,
    quantity: str,
) -> CourtArray:
    _require_court_array(
        value,
        trailing_shape=(dimensions,),
        quantity=quantity,
    )
    if isinstance(value, Tensor):
        rotation = torch.tensor(
            matrix,
            dtype=value.dtype,
            device=value.device,
        )[:dimensions, :dimensions]
        return torch.matmul(value, rotation.transpose(-1, -2))
    rotation_array = np.asarray(matrix, dtype=value.dtype)[:dimensions, :dimensions]
    return np.matmul(value, rotation_array.T)


@overload
def court_points_physical_to_target(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_points_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_points_physical_to_target(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Rotate physical court points into the recorded model target frame."""
    return _rotate_last_dimension(
        value,
        provenance.reference_from_physical,
        dimensions=3,
        quantity="court points",
    )


@overload
def court_points_target_to_physical(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_points_target_to_physical(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_points_target_to_physical(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Invert model target-frame points back to physical court metres."""
    return _rotate_last_dimension(
        value,
        provenance.physical_from_reference,
        dimensions=3,
        quantity="court points",
    )


@overload
def court_vectors_physical_to_target(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_vectors_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_vectors_physical_to_target(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Rotate physical court vectors (for example velocity) into target frame."""
    return _rotate_last_dimension(
        value,
        provenance.reference_from_physical,
        dimensions=3,
        quantity="court vectors",
    )


@overload
def court_vectors_target_to_physical(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_vectors_target_to_physical(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_vectors_target_to_physical(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Invert target-frame court vectors back to the physical court frame."""
    return _rotate_last_dimension(
        value,
        provenance.physical_from_reference,
        dimensions=3,
        quantity="court vectors",
    )


@overload
def court_headings_physical_to_target(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_headings_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_headings_physical_to_target(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Rotate ``(..., 2)`` court heading vectors into target frame."""
    return _rotate_last_dimension(
        value,
        provenance.reference_from_physical,
        dimensions=2,
        quantity="court headings",
    )


@overload
def court_headings_target_to_physical(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_headings_target_to_physical(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_headings_target_to_physical(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Invert target-frame ``(..., 2)`` headings to physical court frame."""
    return _rotate_last_dimension(
        value,
        provenance.physical_from_reference,
        dimensions=2,
        quantity="court headings",
    )


@overload
def court_world_joints_physical_to_target(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_world_joints_physical_to_target(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_world_joints_physical_to_target(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Rotate court-space world joints; player-local pose is not accepted here."""
    return court_points_physical_to_target(value, provenance)


@overload
def court_world_joints_target_to_physical(
    value: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> np.ndarray: ...


@overload
def court_world_joints_target_to_physical(
    value: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> Tensor: ...


def court_world_joints_target_to_physical(
    value: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> CourtArray:
    """Invert court-space world joints back to physical court metres."""
    return court_points_target_to_physical(value, provenance)


def _require_camera_extrinsics(
    camera_center: CourtArray,
    rotation_camera_from_court: CourtArray,
) -> None:
    _require_court_array(
        camera_center,
        trailing_shape=(3,),
        quantity="camera center",
    )
    _require_court_array(
        rotation_camera_from_court,
        trailing_shape=(3, 3),
        quantity="camera rotation",
    )
    if isinstance(camera_center, Tensor) != isinstance(
        rotation_camera_from_court, Tensor
    ):
        raise TypeError(
            "camera center and rotation must both be NumPy arrays or both be "
            "torch tensors."
        )
    if isinstance(camera_center, Tensor):
        assert isinstance(rotation_camera_from_court, Tensor)
        if (
            camera_center.dtype != rotation_camera_from_court.dtype
            or camera_center.device != rotation_camera_from_court.device
        ):
            raise CourtReferenceFrameError(
                "camera center and rotation must share dtype and device."
            )
    else:
        assert isinstance(rotation_camera_from_court, np.ndarray)
        if camera_center.dtype != rotation_camera_from_court.dtype:
            raise CourtReferenceFrameError(
                "camera center and rotation must share dtype."
            )


@overload
def camera_extrinsics_physical_to_target(
    camera_center_physical: np.ndarray,
    rotation_camera_from_physical: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def camera_extrinsics_physical_to_target(
    camera_center_physical: Tensor,
    rotation_camera_from_physical: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> tuple[Tensor, Tensor]: ...


def camera_extrinsics_physical_to_target(
    camera_center_physical: CourtArray,
    rotation_camera_from_physical: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> tuple[CourtArray, CourtArray]:
    """Apply ``C_t=S C_p`` and ``R_cam<-t=R_cam<-p S^T``."""
    _require_camera_extrinsics(
        camera_center_physical,
        rotation_camera_from_physical,
    )
    center_target = court_points_physical_to_target(
        camera_center_physical,
        provenance,
    )
    if isinstance(rotation_camera_from_physical, Tensor):
        rotation = torch.tensor(
            provenance.reference_from_physical,
            dtype=rotation_camera_from_physical.dtype,
            device=rotation_camera_from_physical.device,
        )
        rotation_target = torch.matmul(
            rotation_camera_from_physical,
            rotation.transpose(-1, -2),
        )
    else:
        rotation = np.asarray(
            provenance.reference_from_physical,
            dtype=rotation_camera_from_physical.dtype,
        )
        rotation_target = np.matmul(rotation_camera_from_physical, rotation.T)
    return center_target, rotation_target


@overload
def camera_extrinsics_target_to_physical(
    camera_center_target: np.ndarray,
    rotation_camera_from_target: np.ndarray,
    provenance: CourtReferenceFrameProvenance,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def camera_extrinsics_target_to_physical(
    camera_center_target: Tensor,
    rotation_camera_from_target: Tensor,
    provenance: CourtReferenceFrameProvenance,
) -> tuple[Tensor, Tensor]: ...


def camera_extrinsics_target_to_physical(
    camera_center_target: CourtArray,
    rotation_camera_from_target: CourtArray,
    provenance: CourtReferenceFrameProvenance,
) -> tuple[CourtArray, CourtArray]:
    """Invert target-frame camera centre/rotation to physical court frame."""
    _require_camera_extrinsics(camera_center_target, rotation_camera_from_target)
    center_physical = court_points_target_to_physical(
        camera_center_target,
        provenance,
    )
    if isinstance(rotation_camera_from_target, Tensor):
        rotation = torch.tensor(
            provenance.reference_from_physical,
            dtype=rotation_camera_from_target.dtype,
            device=rotation_camera_from_target.device,
        )
        rotation_physical = torch.matmul(rotation_camera_from_target, rotation)
    else:
        rotation = np.asarray(
            provenance.reference_from_physical,
            dtype=rotation_camera_from_target.dtype,
        )
        rotation_physical = np.matmul(rotation_camera_from_target, rotation)
    return center_physical, rotation_physical
