"""Checkpoint metadata tests for court-coordinate normalization."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.tasks.base.data.court_coordinate_contract import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
    CourtCoordinateContractMismatchError,
    CourtCoordinateNormalizationMetadata,
    InvalidCourtCoordinateMetadataError,
    MissingCourtCoordinateMetadataError,
)
from src.tasks.base.model_io.court_coordinate_contract import (
    resolve_checkpoint_court_coordinate_contract,
    validate_checkpoint_court_coordinate_contract,
    write_checkpoint_court_coordinate_contract,
)
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


def _checkpoint(version: str) -> dict[str, object]:
    metadata = CourtCoordinateNormalizationMetadata.from_contract(
        resolve_court_coordinate_normalization(version)
    )
    return {COURT_COORDINATE_NORMALIZATION_METADATA_KEY: metadata.to_dict()}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_checkpoint_round_trip_restores_its_recorded_contract(version: str) -> None:
    checkpoint: dict[str, object] = {}
    contract = resolve_court_coordinate_normalization(version)
    write_checkpoint_court_coordinate_contract(checkpoint, contract)

    restored = resolve_checkpoint_court_coordinate_contract(checkpoint)

    assert restored.contract == contract
    assert restored.metadata is not None
    assert restored.legacy_metadata_free is False


def test_metadata_free_checkpoint_requires_explicit_v1_runtime() -> None:
    with pytest.raises(MissingCourtCoordinateMetadataError, match="Supply an explicit v1"):
        resolve_checkpoint_court_coordinate_contract({})
    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        validate_checkpoint_court_coordinate_contract(
            {},
            resolve_court_coordinate_normalization("v2"),
        )

    restored = resolve_checkpoint_court_coordinate_contract(
        {},
        legacy_runtime_contract=resolve_court_coordinate_normalization("v1"),
    )
    assert restored.legacy_metadata_free is True


@pytest.mark.parametrize(("stored", "runtime"), [("v1", "v2"), ("v2", "v1")])
def test_checkpoint_runtime_mismatch_is_rejected(stored: str, runtime: str) -> None:
    with pytest.raises(CourtCoordinateContractMismatchError, match="does not match runtime"):
        validate_checkpoint_court_coordinate_contract(
            _checkpoint(stored),
            resolve_court_coordinate_normalization(runtime),
        )


def test_checkpoint_scale_mismatch_is_rejected_even_when_version_matches() -> None:
    checkpoint = _checkpoint("v2")
    raw = deepcopy(checkpoint[COURT_COORDINATE_NORMALIZATION_METADATA_KEY])
    assert isinstance(raw, dict)
    raw["scale_xyz"] = [5.485, 11.885, 1.07]
    checkpoint[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = raw

    with pytest.raises(InvalidCourtCoordinateMetadataError, match="scale_xyz"):
        validate_checkpoint_court_coordinate_contract(
            checkpoint,
            resolve_court_coordinate_normalization("v2"),
        )


def test_checkpoint_write_refuses_to_replace_a_different_version() -> None:
    checkpoint = _checkpoint("v1")

    with pytest.raises(CourtCoordinateContractMismatchError, match="refusing to replace"):
        write_checkpoint_court_coordinate_contract(
            checkpoint,
            resolve_court_coordinate_normalization("v2"),
        )
