"""Fail-closed runtime/direct/checkpoint CourtKP20 contract tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_V2_SELECTOR,
    COURT_KEYPOINT_METADATA_KEY,
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    InvalidCourtKeypointMetadataError,
    MissingCourtKeypointMetadataError,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io.court_keypoint_contract import (
    resolve_model_artifact_court_keypoint_contract,
    validate_model_artifact_court_keypoint_contract,
    write_model_artifact_court_keypoint_contract,
)


def _contract(selector: str) -> CourtKeypointContract:
    return resolve_court_keypoint_contract(selector)


@pytest.mark.parametrize(
    "selector",
    [PHYSICAL_V1_SELECTOR, CAMERA_VIEW_V2_SELECTOR],
)
def test_model_artifact_contract_round_trip(selector: str) -> None:
    document: dict[str, object] = {}
    contract = _contract(selector)
    write_model_artifact_court_keypoint_contract(document, contract)

    resolved = resolve_model_artifact_court_keypoint_contract(document)
    validated = validate_model_artifact_court_keypoint_contract(
        document,
        contract,
    )

    assert resolved.contract == contract
    assert resolved.metadata is not None
    assert resolved.legacy_metadata_free is False
    assert validated == resolved


def test_metadata_free_requires_explicit_physical_v1_runtime() -> None:
    with pytest.raises(MissingCourtKeypointMetadataError, match="explicit physical_v1"):
        resolve_model_artifact_court_keypoint_contract({})
    with pytest.raises(MissingCourtKeypointMetadataError, match="physical_v1"):
        validate_model_artifact_court_keypoint_contract(
            {},
            _contract(CAMERA_VIEW_V2_SELECTOR),
        )
    with pytest.raises(MissingCourtKeypointMetadataError, match="physical_v1"):
        resolve_model_artifact_court_keypoint_contract(
            {},
            explicit_legacy_runtime=_contract(CAMERA_VIEW_V2_SELECTOR),
        )

    legacy = resolve_model_artifact_court_keypoint_contract(
        {},
        explicit_legacy_runtime=_contract(PHYSICAL_V1_SELECTOR),
    )
    assert legacy.contract == _contract(PHYSICAL_V1_SELECTOR)
    assert legacy.metadata is None
    assert legacy.legacy_metadata_free is True


@pytest.mark.parametrize(
    ("stored", "runtime"),
    [
        (PHYSICAL_V1_SELECTOR, CAMERA_VIEW_V2_SELECTOR),
        (CAMERA_VIEW_V2_SELECTOR, PHYSICAL_V1_SELECTOR),
    ],
)
def test_same_shape_semantic_mismatch_is_rejected(
    stored: str,
    runtime: str,
) -> None:
    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(document, _contract(stored))

    with pytest.raises(CourtKeypointContractMismatchError, match="exactly match"):
        validate_model_artifact_court_keypoint_contract(
            document,
            _contract(runtime),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("contract_id", "camera_view_courtkp20_v2"),
        ("target_frame_id", "physical_court_v1"),
        ("num_keypoints", 14),
    ],
)
def test_unknown_or_malformed_contract_metadata_is_rejected(
    field: str,
    value: object,
) -> None:
    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(
        document,
        _contract(CAMERA_VIEW_V2_SELECTOR),
    )
    metadata = deepcopy(document[COURT_KEYPOINT_METADATA_KEY])
    assert isinstance(metadata, dict)
    metadata[field] = value
    document[COURT_KEYPOINT_METADATA_KEY] = metadata

    with pytest.raises(InvalidCourtKeypointMetadataError, match=field):
        validate_model_artifact_court_keypoint_contract(
            document,
            _contract(CAMERA_VIEW_V2_SELECTOR),
        )


def test_write_refuses_to_replace_an_existing_contract() -> None:
    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(
        document,
        _contract(PHYSICAL_V1_SELECTOR),
    )

    with pytest.raises(CourtKeypointContractMismatchError, match="refusing"):
        write_model_artifact_court_keypoint_contract(
            document,
            _contract(CAMERA_VIEW_V2_SELECTOR),
        )
