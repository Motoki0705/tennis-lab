"""Fail-closed runtime/checkpoint compatibility for CourtKP20 semantics."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from src.tasks.base.generate_dataset.court_view import (
    COURT_KEYPOINT_METADATA_KEY,
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMetadata,
    CourtKeypointContractMismatchError,
    MissingCourtKeypointMetadataError,
    extract_court_keypoint_contract_metadata,
    resolve_court_keypoint_contract,
)

__all__ = [
    "ModelArtifactCourtKeypointContract",
    "extract_model_artifact_court_keypoint_contract",
    "resolve_model_artifact_court_keypoint_contract",
    "validate_model_artifact_court_keypoint_contract",
    "write_model_artifact_court_keypoint_contract",
]


@dataclass(frozen=True, slots=True)
class ModelArtifactCourtKeypointContract:
    """Successful direct-input/runtime/checkpoint compatibility result."""

    contract: CourtKeypointContract
    metadata: CourtKeypointContractMetadata | None
    legacy_metadata_free: bool


def extract_model_artifact_court_keypoint_contract(
    document: Mapping[str, object],
    *,
    location: str = "model artifact",
) -> CourtKeypointContractMetadata | None:
    """Extract exact contract metadata without guessing from tensor shapes."""
    return extract_court_keypoint_contract_metadata(document, location=location)


def validate_model_artifact_court_keypoint_contract(
    document: Mapping[str, object],
    runtime_contract: CourtKeypointContract,
    *,
    location: str = "model artifact",
) -> ModelArtifactCourtKeypointContract:
    """Validate runtime/direct/checkpoint metadata before tensors are consumed.

    A metadata-free document is accepted only when the caller has explicitly
    resolved and supplied ``physical_v1``.  Camera-view v2 and every mismatch
    fail closed even though both contracts use 20 Court slots.
    """
    canonical_runtime = resolve_court_keypoint_contract(runtime_contract.selector)
    if runtime_contract != canonical_runtime:
        raise CourtKeypointContractMismatchError(
            f"{location}: runtime court keypoint contract is not canonical: "
            f"{runtime_contract!r}."
        )
    expected = CourtKeypointContractMetadata.from_contract(runtime_contract)
    metadata = extract_model_artifact_court_keypoint_contract(
        document,
        location=location,
    )
    if metadata is None:
        if runtime_contract.selector != PHYSICAL_V1_SELECTOR:
            raise MissingCourtKeypointMetadataError(
                f"{location}: court keypoint metadata is absent. Metadata-free "
                "model artifacts are accepted only by an explicitly selected "
                f"{PHYSICAL_V1_SELECTOR!r} runtime; got "
                f"{runtime_contract.selector!r}."
            )
        return ModelArtifactCourtKeypointContract(
            contract=runtime_contract,
            metadata=None,
            legacy_metadata_free=True,
        )
    if metadata != expected:
        raise CourtKeypointContractMismatchError(
            f"{location}: stored court keypoint contract {metadata.to_dict()!r} "
            f"does not exactly match runtime {expected.to_dict()!r}."
        )
    return ModelArtifactCourtKeypointContract(
        contract=runtime_contract,
        metadata=metadata,
        legacy_metadata_free=False,
    )


def resolve_model_artifact_court_keypoint_contract(
    document: Mapping[str, object],
    *,
    explicit_legacy_runtime: CourtKeypointContract | None = None,
    location: str = "model artifact",
) -> ModelArtifactCourtKeypointContract:
    """Restore recorded semantics, or require explicit physical-v1 for legacy.

    ``explicit_legacy_runtime`` is consulted only when metadata is absent.  It
    must be the canonical physical-v1 contract; passing ``None`` or camera-view
    v2 is an error rather than a fallback.
    """
    metadata = extract_model_artifact_court_keypoint_contract(
        document,
        location=location,
    )
    if metadata is not None:
        return ModelArtifactCourtKeypointContract(
            contract=metadata.contract,
            metadata=metadata,
            legacy_metadata_free=False,
        )
    if explicit_legacy_runtime is None:
        raise MissingCourtKeypointMetadataError(
            f"{location}: cannot resolve court keypoint semantics because "
            "metadata is absent. Supply an explicit physical_v1 runtime only "
            "for a known legacy artifact."
        )
    return validate_model_artifact_court_keypoint_contract(
        document,
        explicit_legacy_runtime,
        location=location,
    )


def write_model_artifact_court_keypoint_contract(
    document: MutableMapping[str, object],
    contract: CourtKeypointContract,
    *,
    location: str = "model artifact",
) -> None:
    """Persist canonical contract metadata without replacing a conflict."""
    expected = CourtKeypointContractMetadata.from_contract(contract)
    existing = extract_model_artifact_court_keypoint_contract(
        document,
        location=location,
    )
    if existing is not None and existing != expected:
        raise CourtKeypointContractMismatchError(
            f"{location}: refusing to replace stored court keypoint contract "
            f"{existing.to_dict()!r} with {expected.to_dict()!r}."
        )
    document[COURT_KEYPOINT_METADATA_KEY] = expected.to_dict()
