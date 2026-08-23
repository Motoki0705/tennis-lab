"""Adversarial dataset metadata tests for the court-coordinate contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
    CourtCoordinateContractMismatchError,
    CourtCoordinateNormalizationMetadata,
    InvalidCourtCoordinateMetadataError,
    MissingCourtCoordinateMetadataError,
    MixedCourtCoordinateMetadataError,
    inject_court_coordinate_normalization_metadata,
    validate_dataset_court_coordinate_contract,
    validate_dataset_court_coordinate_contract_documents,
)
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


def _metadata(version: str) -> dict[str, object]:
    return cast(
        dict[str, object],
        CourtCoordinateNormalizationMetadata.from_contract(
            resolve_court_coordinate_normalization(version)
        ).to_dict(),
    )


def _document(version: str) -> dict[str, object]:
    return {COURT_COORDINATE_NORMALIZATION_METADATA_KEY: _metadata(version)}


def test_new_metadata_records_version_scale_and_physical_units() -> None:
    metadata = _metadata("v2")

    assert metadata == {
        "schema_version": 1,
        "version": "v2",
        "scale_xyz": [11.885, 11.885, 11.885],
        "position_unit": "m",
        "velocity_unit": "m/s",
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_identical_root_and_scene_contract_is_accepted(version: str) -> None:
    contract = resolve_court_coordinate_normalization(version)
    result = validate_dataset_court_coordinate_contract_documents(
        root_metadata=_document(version),
        scene_metadata={"scene_a": _document(version), "scene_b": _document(version)},
        runtime_contract=contract,
    )

    assert result.contract == contract
    assert result.metadata is not None
    assert result.legacy_metadata_free is False
    assert result.scene_count == 2


def test_all_missing_metadata_is_legacy_v1_only() -> None:
    legacy = validate_dataset_court_coordinate_contract_documents(
        root_metadata={},
        scene_metadata={"scene_a": {}, "scene_b": {}},
        runtime_contract=resolve_court_coordinate_normalization("v1"),
    )
    assert legacy.legacy_metadata_free is True

    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        validate_dataset_court_coordinate_contract_documents(
            root_metadata={},
            scene_metadata={"scene_a": {}, "scene_b": {}},
            runtime_contract=resolve_court_coordinate_normalization("v2"),
        )


@pytest.mark.parametrize(
    ("root", "scenes"),
    [
        (_document("v1"), {"scene_a": {}}),
        ({}, {"scene_a": _document("v1")}),
        (_document("v1"), {"scene_a": _document("v1"), "scene_b": {}}),
    ],
)
def test_partial_root_scene_metadata_is_rejected(
    root: dict[str, object],
    scenes: dict[str, dict[str, object]],
) -> None:
    with pytest.raises(MixedCourtCoordinateMetadataError, match="mixed"):
        validate_dataset_court_coordinate_contract_documents(
            root_metadata=root,
            scene_metadata=scenes,
            runtime_contract=resolve_court_coordinate_normalization("v1"),
        )


@pytest.mark.parametrize(
    ("runtime", "root_version", "scene_version"),
    [
        ("v1", "v2", "v2"),
        ("v2", "v1", "v1"),
        ("v2", "v2", "v1"),
        ("v1", "v1", "v2"),
    ],
)
def test_runtime_root_scene_version_mismatch_is_rejected(
    runtime: str,
    root_version: str,
    scene_version: str,
) -> None:
    with pytest.raises(CourtCoordinateContractMismatchError, match="does not match runtime"):
        validate_dataset_court_coordinate_contract_documents(
            root_metadata=_document(root_version),
            scene_metadata={"scene_a": _document(scene_version)},
            runtime_contract=resolve_court_coordinate_normalization(runtime),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "v3"),
        ("scale_xyz", [11.885, 11.885, 1.07]),
        ("position_unit", "cm"),
        ("velocity_unit", "normalized/s"),
        ("schema_version", 2),
    ],
)
def test_unknown_or_malformed_metadata_is_rejected(
    field: str,
    value: object,
) -> None:
    document = _document("v2")
    metadata = deepcopy(document[COURT_COORDINATE_NORMALIZATION_METADATA_KEY])
    assert isinstance(metadata, dict)
    metadata[field] = value
    document[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = metadata

    with pytest.raises(InvalidCourtCoordinateMetadataError, match=field):
        validate_dataset_court_coordinate_contract_documents(
            root_metadata=document,
            scene_metadata={"scene_a": _document("v2")},
            runtime_contract=resolve_court_coordinate_normalization("v2"),
        )


def test_injection_refuses_to_replace_an_existing_contract() -> None:
    with pytest.raises(CourtCoordinateContractMismatchError, match="does not match"):
        inject_court_coordinate_normalization_metadata(
            _document("v1"),
            resolve_court_coordinate_normalization("v2"),
            location="dataset/meta.json",
        )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_metadata_free_v1_root_without_root_meta_json_is_supported(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scenes" / "scene_000001"
    _write_json(scene / "meta.json", {"scene_id": "scene_000001"})

    result = validate_dataset_court_coordinate_contract(
        tmp_path,
        resolve_court_coordinate_normalization("v1"),
    )

    assert result.legacy_metadata_free is True
    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        validate_dataset_court_coordinate_contract(
            tmp_path,
            resolve_court_coordinate_normalization("v2"),
        )
