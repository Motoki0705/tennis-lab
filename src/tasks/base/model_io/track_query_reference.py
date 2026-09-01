"""Fail-closed track-query runtime/checkpoint compatibility metadata.

Court semantics, target frame, spatial RoPE semantics, and selector mode are
stored as independent fields.  Matching parameter shapes never substitute for
these markers.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Final, cast

from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID,
    PHYSICAL_COURT_TARGET_FRAME_ID,
    PHYSICAL_COURTKP20_CONTRACT_ID,
    REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID,
    CourtKeypointContractId,
    CourtTargetFrameId,
    resolve_court_keypoint_contract_id,
)
from src.tasks.base.models.track_query_reference import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    ROLE_ROPE_CONTRACT,
    ReferenceSelectorMode,
    TrackQueryRopeContract,
    resolve_reference_selector_mode,
    resolve_track_query_rope_contract,
)

TRACK_QUERY_REFERENCE_METADATA_KEY: Final = "track_query_reference"
TRACK_QUERY_REFERENCE_METADATA_SCHEMA_VERSION: Final = 1
TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY: Final = (
    "track_query_checkpoint_architecture"
)
TRACK_QUERY_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION: Final = 1
TRACK_QUERY_CHECKPOINT_ARCHITECTURE_ID: Final = "fixed_query_track_compressed_v1"
REFERENCE_SELECTOR_NOT_APPLICABLE: Final = "not_applicable"

_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "court_keypoint_contract",
        "target_frame_contract",
        "track_query_rope_contract",
        "reference_selector_mode",
    }
)
_CHECKPOINT_ARCHITECTURE_METADATA_FIELDS = frozenset(
    {"schema_version", "architecture"}
)


class TrackQueryReferenceContractError(ValueError):
    """Base compatibility/checkpoint contract error."""


class MissingTrackQueryReferenceMetadataError(TrackQueryReferenceContractError):
    """Raised when required semantic metadata is absent."""


class InvalidTrackQueryReferenceMetadataError(TrackQueryReferenceContractError):
    """Raised when persisted semantic metadata is malformed or unknown."""


class TrackQueryReferenceContractMismatchError(TrackQueryReferenceContractError):
    """Raised when explicit runtime and artifact contracts differ."""


@dataclass(frozen=True, slots=True)
class TrackQueryReferenceContract:
    """Independent runtime semantics for one track-query model family."""

    court_keypoint_contract: CourtKeypointContractId
    target_frame_contract: CourtTargetFrameId
    track_query_rope_contract: TrackQueryRopeContract
    reference_selector_mode: ReferenceSelectorMode | None

    def __post_init__(self) -> None:
        try:
            resolve_court_keypoint_contract_id(self.court_keypoint_contract)
        except ValueError as error:
            raise TrackQueryReferenceContractError(str(error)) from error
        expected: tuple[
            CourtKeypointContractId,
            CourtTargetFrameId,
            ReferenceSelectorMode | None,
        ]
        if self.track_query_rope_contract is ROLE_ROPE_CONTRACT:
            expected = (
                PHYSICAL_COURTKP20_CONTRACT_ID,
                PHYSICAL_COURT_TARGET_FRAME_ID,
                None,
            )
        elif self.track_query_rope_contract is REFERENCE_SELECTOR_ROPE_CONTRACT:
            if not isinstance(self.reference_selector_mode, ReferenceSelectorMode):
                raise TrackQueryReferenceContractError(
                    "Reference-selector v2 requires explicit reference mode."
                )
            expected = (
                CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID,
                REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID,
                self.reference_selector_mode,
            )
        else:
            raise TrackQueryReferenceContractError(
                "track_query_rope_contract must be TrackQueryRopeContract."
            )
        actual = (
            self.court_keypoint_contract,
            self.target_frame_contract,
            self.reference_selector_mode,
        )
        if actual != expected:
            raise TrackQueryReferenceContractError(
                f"RoPE contract {self.track_query_rope_contract.value!r} requires "
                f"court/target/selector {expected!r}; got {actual!r}."
            )

    @classmethod
    def physical_v1(cls) -> TrackQueryReferenceContract:
        """Return canonical physical-court role-axis and five-input semantics."""
        return cls(
            court_keypoint_contract=PHYSICAL_COURTKP20_CONTRACT_ID,
            target_frame_contract=PHYSICAL_COURT_TARGET_FRAME_ID,
            track_query_rope_contract=ROLE_ROPE_CONTRACT,
            reference_selector_mode=None,
        )

    @classmethod
    def reference_v2(
        cls,
        selector_mode: ReferenceSelectorMode,
    ) -> TrackQueryReferenceContract:
        """Return explicit camera-view/reference-frame six-input semantics."""
        if not isinstance(selector_mode, ReferenceSelectorMode):
            raise TypeError("selector_mode must be ReferenceSelectorMode.")
        return cls(
            court_keypoint_contract=CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID,
            target_frame_contract=REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID,
            track_query_rope_contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
            reference_selector_mode=selector_mode,
        )


@dataclass(frozen=True, slots=True)
class TrackQueryReferenceContractMetadata:
    """Exact JSON record for a validated runtime contract."""

    schema_version: int
    contract: TrackQueryReferenceContract

    @classmethod
    def from_contract(
        cls,
        contract: TrackQueryReferenceContract,
    ) -> TrackQueryReferenceContractMetadata:
        """Build canonical metadata after revalidating all independent fields."""
        canonical = TrackQueryReferenceContract(
            court_keypoint_contract=contract.court_keypoint_contract,
            target_frame_contract=contract.target_frame_contract,
            track_query_rope_contract=contract.track_query_rope_contract,
            reference_selector_mode=contract.reference_selector_mode,
        )
        return cls(
            schema_version=TRACK_QUERY_REFERENCE_METADATA_SCHEMA_VERSION,
            contract=canonical,
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        location: str,
    ) -> TrackQueryReferenceContractMetadata:
        """Parse strict metadata without inference from other checkpoint keys."""
        if not isinstance(value, Mapping):
            raise InvalidTrackQueryReferenceMetadataError(
                f"{location}: expected a metadata mapping."
            )
        mapping = cast("Mapping[object, object]", value)
        if set(mapping) != _METADATA_FIELDS:
            missing = sorted(str(field) for field in _METADATA_FIELDS - set(mapping))
            unknown = sorted(str(field) for field in set(mapping) - _METADATA_FIELDS)
            raise InvalidTrackQueryReferenceMetadataError(
                f"{location}: invalid metadata fields; missing={missing!r}, "
                f"unknown={unknown!r}."
            )
        schema_version = mapping["schema_version"]
        if (
            type(schema_version) is not int
            or schema_version != TRACK_QUERY_REFERENCE_METADATA_SCHEMA_VERSION
        ):
            raise InvalidTrackQueryReferenceMetadataError(
                f"{location}.schema_version: expected "
                f"{TRACK_QUERY_REFERENCE_METADATA_SCHEMA_VERSION}, got "
                f"{schema_version!r}."
            )
        raw_court = mapping["court_keypoint_contract"]
        raw_target = mapping["target_frame_contract"]
        raw_rope = mapping["track_query_rope_contract"]
        raw_selector = mapping["reference_selector_mode"]
        if any(
            type(value) is not str
            for value in (raw_court, raw_target, raw_rope, raw_selector)
        ):
            raise InvalidTrackQueryReferenceMetadataError(
                f"{location}: all semantic marker fields must be strings."
            )
        try:
            rope = resolve_track_query_rope_contract(cast("str", raw_rope))
            selector = (
                None
                if raw_selector == REFERENCE_SELECTOR_NOT_APPLICABLE
                else resolve_reference_selector_mode(cast("str", raw_selector))
            )
            contract = TrackQueryReferenceContract(
                court_keypoint_contract=cast("CourtKeypointContractId", raw_court),
                target_frame_contract=cast("CourtTargetFrameId", raw_target),
                track_query_rope_contract=rope,
                reference_selector_mode=selector,
            )
        except (TypeError, ValueError) as error:
            raise InvalidTrackQueryReferenceMetadataError(
                f"{location}.court_keypoint_contract/target_frame_contract/"
                "track_query_rope_contract/reference_selector_mode: "
                f"{error}"
            ) from error
        return cls.from_contract(contract)

    def to_dict(self) -> dict[str, object]:
        """Return the exact independent marker mapping."""
        selector = self.contract.reference_selector_mode
        return {
            "schema_version": self.schema_version,
            "court_keypoint_contract": self.contract.court_keypoint_contract,
            "target_frame_contract": self.contract.target_frame_contract,
            "track_query_rope_contract": (
                self.contract.track_query_rope_contract.value
            ),
            "reference_selector_mode": (
                REFERENCE_SELECTOR_NOT_APPLICABLE
                if selector is None
                else selector.value
            ),
        }


@dataclass(frozen=True, slots=True)
class ModelArtifactTrackQueryReferenceContract:
    """Successful runtime/direct/checkpoint compatibility result."""

    contract: TrackQueryReferenceContract
    metadata: TrackQueryReferenceContractMetadata


def extract_track_query_reference_contract_metadata(
    document: Mapping[str, object],
    *,
    location: str = "model artifact",
) -> TrackQueryReferenceContractMetadata | None:
    """Extract metadata, returning ``None`` only for an absent top-level key."""
    if TRACK_QUERY_REFERENCE_METADATA_KEY not in document:
        return None
    return TrackQueryReferenceContractMetadata.from_mapping(
        document[TRACK_QUERY_REFERENCE_METADATA_KEY],
        location=f"{location}.{TRACK_QUERY_REFERENCE_METADATA_KEY}",
    )


def validate_track_query_reference_contract(
    document: Mapping[str, object],
    runtime_contract: TrackQueryReferenceContract,
    *,
    location: str = "model artifact",
) -> ModelArtifactTrackQueryReferenceContract:
    """Validate exact runtime/artifact semantics before state or tensors load."""
    expected = TrackQueryReferenceContractMetadata.from_contract(runtime_contract)
    metadata = extract_track_query_reference_contract_metadata(
        document,
        location=location,
    )
    if metadata is None:
        raise MissingTrackQueryReferenceMetadataError(
            f"{location}: track-query semantic metadata is absent. Canonical "
            "track-query artifacts never infer semantics from config or weight "
            "shape."
        )
    if metadata != expected:
        raise TrackQueryReferenceContractMismatchError(
            f"{location}: stored track-query contract {metadata.to_dict()!r} does "
            f"not exactly match runtime {expected.to_dict()!r}."
        )
    return ModelArtifactTrackQueryReferenceContract(
        contract=runtime_contract,
        metadata=metadata,
    )


def resolve_track_query_reference_contract(
    document: Mapping[str, object],
    *,
    location: str = "model artifact",
) -> ModelArtifactTrackQueryReferenceContract:
    """Resolve recorded markers without inferring absent semantics."""
    metadata = extract_track_query_reference_contract_metadata(
        document,
        location=location,
    )
    if metadata is None:
        raise MissingTrackQueryReferenceMetadataError(
            f"{location}: cannot resolve track-query semantics because metadata "
            "is absent."
        )
    return ModelArtifactTrackQueryReferenceContract(
        contract=metadata.contract,
        metadata=metadata,
    )


def _checkpoint_architecture_metadata() -> dict[str, object]:
    return {
        "schema_version": TRACK_QUERY_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION,
        "architecture": TRACK_QUERY_CHECKPOINT_ARCHITECTURE_ID,
    }


def _validate_checkpoint_architecture_metadata(
    checkpoint: Mapping[str, object],
    *,
    location: str,
) -> None:
    if TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY not in checkpoint:
        raise MissingTrackQueryReferenceMetadataError(
            f"{location}: canonical track-query checkpoint architecture metadata "
            "is absent; pre-promotion checkpoints are incompatible."
        )
    value = checkpoint[TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY]
    if not isinstance(value, Mapping):
        raise InvalidTrackQueryReferenceMetadataError(
            f"{location}.{TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY}: "
            "expected a metadata mapping."
        )
    mapping = cast("Mapping[object, object]", value)
    if set(mapping) != _CHECKPOINT_ARCHITECTURE_METADATA_FIELDS:
        missing = sorted(
            str(field)
            for field in _CHECKPOINT_ARCHITECTURE_METADATA_FIELDS - set(mapping)
        )
        unknown = sorted(
            str(field)
            for field in set(mapping) - _CHECKPOINT_ARCHITECTURE_METADATA_FIELDS
        )
        raise InvalidTrackQueryReferenceMetadataError(
            f"{location}.{TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY}: "
            f"invalid metadata fields; missing={missing!r}, unknown={unknown!r}."
        )
    expected = _checkpoint_architecture_metadata()
    if dict(mapping) != expected:
        raise TrackQueryReferenceContractMismatchError(
            f"{location}: stored track-query checkpoint architecture "
            f"{dict(mapping)!r} does not exactly match runtime {expected!r}."
        )


def write_track_query_reference_contract(
    document: MutableMapping[str, object],
    contract: TrackQueryReferenceContract,
    *,
    location: str = "model artifact",
) -> None:
    """Persist canonical markers without replacing any conflicting record."""
    expected = TrackQueryReferenceContractMetadata.from_contract(contract)
    existing = extract_track_query_reference_contract_metadata(
        document,
        location=location,
    )
    if existing is not None and existing != expected:
        raise TrackQueryReferenceContractMismatchError(
            f"{location}: refusing to replace stored track-query contract "
            f"{existing.to_dict()!r} with {expected.to_dict()!r}."
        )
    document[TRACK_QUERY_REFERENCE_METADATA_KEY] = expected.to_dict()


def validate_checkpoint_track_query_reference_contract(
    checkpoint: Mapping[str, object],
    runtime_contract: TrackQueryReferenceContract,
    *,
    location: str = "checkpoint",
) -> ModelArtifactTrackQueryReferenceContract:
    """Require the canonical architecture and exact semantic marker tuple."""
    _validate_checkpoint_architecture_metadata(checkpoint, location=location)
    return validate_track_query_reference_contract(
        checkpoint,
        runtime_contract,
        location=location,
    )


def write_checkpoint_track_query_reference_contract(
    checkpoint: MutableMapping[str, object],
    contract: TrackQueryReferenceContract,
    *,
    location: str = "checkpoint",
) -> None:
    """Persist the canonical architecture and exact semantic marker tuple."""
    if TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY in checkpoint:
        _validate_checkpoint_architecture_metadata(checkpoint, location=location)
    write_track_query_reference_contract(
        checkpoint,
        contract,
        location=location,
    )
    checkpoint[TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY] = (
        _checkpoint_architecture_metadata()
    )


__all__ = [
    "REFERENCE_SELECTOR_NOT_APPLICABLE",
    "TRACK_QUERY_CHECKPOINT_ARCHITECTURE_ID",
    "TRACK_QUERY_CHECKPOINT_ARCHITECTURE_METADATA_KEY",
    "TRACK_QUERY_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION",
    "TRACK_QUERY_REFERENCE_METADATA_KEY",
    "TRACK_QUERY_REFERENCE_METADATA_SCHEMA_VERSION",
    "InvalidTrackQueryReferenceMetadataError",
    "MissingTrackQueryReferenceMetadataError",
    "ModelArtifactTrackQueryReferenceContract",
    "TrackQueryReferenceContract",
    "TrackQueryReferenceContractError",
    "TrackQueryReferenceContractMetadata",
    "TrackQueryReferenceContractMismatchError",
    "extract_track_query_reference_contract_metadata",
    "resolve_track_query_reference_contract",
    "validate_checkpoint_track_query_reference_contract",
    "validate_track_query_reference_contract",
    "write_checkpoint_track_query_reference_contract",
    "write_track_query_reference_contract",
]
