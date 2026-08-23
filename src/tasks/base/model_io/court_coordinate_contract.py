"""Checkpoint persistence and validation for court-coordinate normalization."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
    CourtCoordinateContractMismatchError,
    CourtCoordinateNormalizationMetadata,
    MissingCourtCoordinateMetadataError,
    extract_court_coordinate_normalization_metadata,
)
from src.utils.schema.court_normalization import CourtCoordinateNormalization

__all__ = [
    "CheckpointCourtCoordinateContract",
    "extract_checkpoint_court_coordinate_normalization",
    "resolve_checkpoint_court_coordinate_contract",
    "validate_checkpoint_court_coordinate_contract",
    "write_checkpoint_court_coordinate_contract",
]


@dataclass(frozen=True, slots=True)
class CheckpointCourtCoordinateContract:
    """Successful checkpoint compatibility result."""

    contract: CourtCoordinateNormalization
    metadata: CourtCoordinateNormalizationMetadata | None
    legacy_metadata_free: bool


def extract_checkpoint_court_coordinate_normalization(
    checkpoint: Mapping[str, object],
    *,
    location: str = "checkpoint",
) -> CourtCoordinateNormalizationMetadata | None:
    """Extract canonical root checkpoint metadata without guessing a version."""
    return extract_court_coordinate_normalization_metadata(
        checkpoint,
        location=location,
    )


def validate_checkpoint_court_coordinate_contract(
    checkpoint: Mapping[str, object],
    runtime_contract: CourtCoordinateNormalization,
    *,
    location: str = "checkpoint",
) -> CheckpointCourtCoordinateContract:
    """Validate a checkpoint before weights or saved state are consumed."""
    expected = CourtCoordinateNormalizationMetadata.from_contract(runtime_contract)
    metadata = extract_checkpoint_court_coordinate_normalization(
        checkpoint,
        location=location,
    )
    if metadata is None:
        if runtime_contract.version != "v1":
            raise MissingCourtCoordinateMetadataError(
                f"{location}: normalization metadata is absent. Metadata-free "
                "checkpoints are legacy v1 only, but runtime selected "
                f"{runtime_contract.version!r}."
            )
        return CheckpointCourtCoordinateContract(
            contract=runtime_contract,
            metadata=None,
            legacy_metadata_free=True,
        )
    if metadata != expected:
        raise CourtCoordinateContractMismatchError(
            f"{location}: checkpoint normalization {metadata.to_dict()!r} does "
            f"not match runtime {expected.to_dict()!r}."
        )
    return CheckpointCourtCoordinateContract(
        contract=runtime_contract,
        metadata=metadata,
        legacy_metadata_free=False,
    )


def resolve_checkpoint_court_coordinate_contract(
    checkpoint: Mapping[str, object],
    *,
    legacy_runtime_contract: CourtCoordinateNormalization | None = None,
    location: str = "checkpoint",
) -> CheckpointCourtCoordinateContract:
    """Restore a contract from checkpoint metadata.

    A metadata-free checkpoint has no derivable version. It can be restored
    only when the caller supplies an explicit ``v1`` runtime contract; passing
    ``None`` or ``v2`` fails instead of inferring from weights or values.
    """
    metadata = extract_checkpoint_court_coordinate_normalization(
        checkpoint,
        location=location,
    )
    if metadata is not None:
        return CheckpointCourtCoordinateContract(
            contract=metadata.contract,
            metadata=metadata,
            legacy_metadata_free=False,
        )
    if legacy_runtime_contract is None:
        raise MissingCourtCoordinateMetadataError(
            f"{location}: cannot restore court-coordinate normalization because "
            "metadata is absent. Supply an explicit v1 runtime contract only "
            "for a known legacy checkpoint."
        )
    return validate_checkpoint_court_coordinate_contract(
        checkpoint,
        legacy_runtime_contract,
        location=location,
    )


def write_checkpoint_court_coordinate_contract(
    checkpoint: MutableMapping[str, object],
    contract: CourtCoordinateNormalization,
    *,
    location: str = "checkpoint",
) -> None:
    """Persist canonical metadata without replacing a conflicting contract."""
    expected = CourtCoordinateNormalizationMetadata.from_contract(contract)
    existing = extract_checkpoint_court_coordinate_normalization(
        checkpoint,
        location=location,
    )
    if existing is not None and existing != expected:
        raise CourtCoordinateContractMismatchError(
            f"{location}: refusing to replace checkpoint normalization "
            f"{existing.to_dict()!r} with {expected.to_dict()!r}."
        )
    checkpoint[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = expected.to_dict()
