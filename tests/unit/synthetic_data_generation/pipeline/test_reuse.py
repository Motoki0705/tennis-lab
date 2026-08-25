"""Fail-closed reuse tests for normalized PLCS publications."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCS_COORDINATE_CONTRACT,
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.pipeline.reuse import (
    PLCSV5ReusablePublicationValidator,
)
from src.utils.schema.court_normalization import (
    COURT_COORDINATE_NORMALIZATION_KEY,
    CourtCoordinateContractError,
    court_coordinate_normalization_metadata,
)


def _write_publication(root: Path, *, mutation: str) -> None:
    root.mkdir()
    contract: object = court_coordinate_normalization_metadata()
    if mutation == "malformed":
        contract = "isotropic_half_length"
    elif mutation in {"unknown", "mismatched"}:
        contract = deepcopy(contract)
        assert isinstance(contract, dict)
        if mutation == "unknown":
            contract["identity"] = "anisotropic"
        else:
            contract["scale_xyz_m"] = [5.485, 11.885, 1.07]
    metadata = {
        "coordinate_contract": PLCS_COORDINATE_CONTRACT.to_dict(),
        COURT_COORDINATE_NORMALIZATION_KEY: contract,
        "logical_scenes": [
            {
                "tracks": [
                    {
                        "support_plane": (
                            PLCSSourceSupportPlane.from_surface_minimum(
                                initial_root_translation_z_m=0.0,
                                support_local_z_m=0.0,
                            ).to_dict()
                        )
                    }
                ]
            }
        ],
    }
    if mutation == "missing":
        del metadata[COURT_COORDINATE_NORMALIZATION_KEY]
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": PLCS_DATASET_SCHEMA,
                "domain": "plcs",
                "metadata": metadata,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("mutation", ["missing", "malformed", "unknown", "mismatched"])
def test_plcs_reuse_rejects_incompatible_normalization_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    publication = tmp_path / mutation
    _write_publication(publication, mutation=mutation)

    with pytest.raises(CourtCoordinateContractError, match="incompatible|mismatched"):
        PLCSV5ReusablePublicationValidator().validate(publication)


def test_plcs_reuse_accepts_the_exact_current_normalization_contract(
    tmp_path: Path,
) -> None:
    publication = tmp_path / "current"
    _write_publication(publication, mutation="valid")

    PLCSV5ReusablePublicationValidator().validate(publication)


def test_plcs_reuse_checks_normalization_before_downstream_scene_metadata(
    tmp_path: Path,
) -> None:
    publication = tmp_path / "mixed-invalid"
    _write_publication(publication, mutation="mismatched")
    manifest_path = publication / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["coordinate_contract"] = {}
    manifest["metadata"]["logical_scenes"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CourtCoordinateContractError, match="mismatched"):
        PLCSV5ReusablePublicationValidator().validate(publication)
