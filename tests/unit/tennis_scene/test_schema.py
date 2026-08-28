"""Tests for CourtKP20 provenance in the canonical scene schema."""

from __future__ import annotations

import pytest

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    build_court_view_record,
    build_physical_court_provenance,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tennis_scene.schema import (
    COURT_REFERENCE_PROVENANCE_KEY,
    attach_court_keypoint_provenance,
    validate_court_keypoint_provenance,
)


def _camera_view_provenance() -> tuple[
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    view = build_court_view_record(
        camera_id="far",
        camera_center_court_m=(2.0, 12.0, 4.0),
        contract=contract,
    )
    return contract, build_reference_frame_provenance(
        (view,),
        reference_camera_id="far",
    )


def test_attach_and_validate_camera_view_provenance_round_trip() -> None:
    contract, provenance = _camera_view_provenance()
    original = {"payload": [1, 2, 3]}

    document = attach_court_keypoint_provenance(
        original,
        contract,
        provenance,
        location="fixture",
    )

    assert original == {"payload": [1, 2, 3]}
    assert document[COURT_REFERENCE_PROVENANCE_KEY] == provenance.to_dict()
    assert (
        validate_court_keypoint_provenance(
            document,
            contract,
            location="fixture",
        )
        == provenance
    )


def test_attach_rejects_provenance_from_another_contract() -> None:
    _, camera_view_provenance = _camera_view_provenance()
    physical_contract = resolve_court_keypoint_contract("physical_v1")

    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        attach_court_keypoint_provenance(
            {},
            physical_contract,
            camera_view_provenance,
            location="fixture",
        )


def test_validate_camera_view_requires_explicit_provenance() -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(document, contract)

    with pytest.raises(
        MissingCourtKeypointMetadataError,
        match="camera_view_v2 result is missing",
    ):
        validate_court_keypoint_provenance(document, contract, location="fixture")


def test_validate_metadata_free_physical_v1_uses_identity_provenance() -> None:
    contract = resolve_court_keypoint_contract("physical_v1")

    provenance = validate_court_keypoint_provenance(
        {},
        contract,
        location="legacy fixture",
    )

    assert provenance == build_physical_court_provenance()
