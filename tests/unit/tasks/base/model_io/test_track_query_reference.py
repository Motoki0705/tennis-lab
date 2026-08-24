"""Independent track-query runtime/checkpoint marker tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.tasks.base.model_io.track_query_reference import (
    TRACK_QUERY_REFERENCE_METADATA_KEY,
    InvalidTrackQueryReferenceMetadataError,
    MissingTrackQueryReferenceMetadataError,
    TrackQueryReferenceContract,
    TrackQueryReferenceContractError,
    TrackQueryReferenceContractMismatchError,
    resolve_track_query_reference_contract,
    validate_checkpoint_track_query_reference_contract,
    validate_track_query_reference_contract,
    write_checkpoint_track_query_reference_contract,
    write_track_query_reference_contract,
)
from src.tasks.base.models.track_query_reference import (
    REFERENCE_SELECTOR_ROPE_CONTRACT,
    ReferenceSelectorMode,
)


@pytest.mark.parametrize(
    "contract",
    [
        TrackQueryReferenceContract.legacy_v1(),
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.SELECTOR_ZERO),
    ],
)
def test_independent_metadata_round_trip(
    contract: TrackQueryReferenceContract,
) -> None:
    document: dict[str, object] = {}
    write_track_query_reference_contract(document, contract)

    result = validate_track_query_reference_contract(document, contract)
    resolved = resolve_track_query_reference_contract(document)

    assert result == resolved
    assert result.contract == contract
    assert result.metadata is not None
    assert result.legacy_metadata_free is False
    metadata = document[TRACK_QUERY_REFERENCE_METADATA_KEY]
    assert isinstance(metadata, dict)
    assert set(metadata) == {
        "schema_version",
        "court_keypoint_contract",
        "target_frame_contract",
        "track_query_rope_contract",
        "reference_selector_mode",
    }


def test_metadata_free_requires_explicit_legacy_v1_and_rejects_v2() -> None:
    with pytest.raises(MissingTrackQueryReferenceMetadataError, match="absent"):
        resolve_track_query_reference_contract({})
    with pytest.raises(MissingTrackQueryReferenceMetadataError, match="legacy v1"):
        validate_track_query_reference_contract(
            {},
            TrackQueryReferenceContract.legacy_v1(),
        )
    with pytest.raises(MissingTrackQueryReferenceMetadataError, match="never infers"):
        validate_track_query_reference_contract(
            {},
            TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
            explicit_legacy_v1=True,
        )

    legacy = resolve_track_query_reference_contract({}, explicit_legacy_v1=True)
    assert legacy.contract == TrackQueryReferenceContract.legacy_v1()
    assert legacy.metadata is None
    assert legacy.legacy_metadata_free is True


@pytest.mark.parametrize(
    ("stored", "runtime"),
    [
        (
            TrackQueryReferenceContract.legacy_v1(),
            TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
        ),
        (
            TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
            TrackQueryReferenceContract.reference_v2(
                ReferenceSelectorMode.SELECTOR_ZERO
            ),
        ),
    ],
)
def test_shape_compatible_semantic_mismatch_is_rejected(
    stored: TrackQueryReferenceContract,
    runtime: TrackQueryReferenceContract,
) -> None:
    checkpoint: dict[str, object] = {}
    write_checkpoint_track_query_reference_contract(checkpoint, stored)
    checkpoint["state_dict"] = {"same_shape": [1, 2, 3]}

    with pytest.raises(TrackQueryReferenceContractMismatchError, match="exactly match"):
        validate_checkpoint_track_query_reference_contract(checkpoint, runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("court_keypoint_contract", "unknown_court"),
        ("target_frame_contract", "physical_court_v1"),
        ("track_query_rope_contract", "v2"),
        ("reference_selector_mode", "role_rope_enabled"),
    ],
)
def test_missing_unknown_and_mixed_independent_markers_fail_closed(
    field: str,
    value: object,
) -> None:
    document: dict[str, object] = {}
    contract = TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE)
    write_track_query_reference_contract(document, contract)
    metadata = deepcopy(document[TRACK_QUERY_REFERENCE_METADATA_KEY])
    assert isinstance(metadata, dict)
    metadata[field] = value
    document[TRACK_QUERY_REFERENCE_METADATA_KEY] = metadata

    with pytest.raises(InvalidTrackQueryReferenceMetadataError, match=field):
        validate_track_query_reference_contract(document, contract)


def test_runtime_contract_rejects_mixed_tuple_before_checkpoint_load() -> None:
    with pytest.raises(TrackQueryReferenceContractError, match="requires"):
        TrackQueryReferenceContract(
            court_keypoint_contract="physical_courtkp20_v1",
            target_frame_contract="physical_court_v1",
            track_query_rope_contract=REFERENCE_SELECTOR_ROPE_CONTRACT,
            reference_selector_mode=ReferenceSelectorMode.REFERENCE,
        )


def test_write_refuses_to_replace_an_existing_selector_mode() -> None:
    document: dict[str, object] = {}
    write_track_query_reference_contract(
        document,
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
    )
    with pytest.raises(TrackQueryReferenceContractMismatchError, match="refusing"):
        write_track_query_reference_contract(
            document,
            TrackQueryReferenceContract.reference_v2(
                ReferenceSelectorMode.SELECTOR_ZERO
            ),
        )
