"""Tests for the exact PLCS AMASS and support-plane coordinate contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCS_COORDINATE_CONTRACT,
    PLCS_COORDINATE_CONTRACT_SCHEMA,
    PLCS_SUPPORT_PLACEMENT_TOLERANCE_M,
    PLCS_SUPPORT_PLANE_SCHEMA,
    SMPLH_SURFACE_VERTEX_COUNT,
    PLCSCoordinateContract,
    PLCSSourceSupportPlane,
)


def test_coordinate_contract_round_trips_only_the_exact_z_up_identity() -> None:
    payload = PLCS_COORDINATE_CONTRACT.to_dict()

    parsed = PLCSCoordinateContract.from_dict(payload)

    assert parsed == PLCS_COORDINATE_CONTRACT
    assert payload == {
        "schema": PLCS_COORDINATE_CONTRACT_SCHEMA,
        "handedness": "right-handed",
        "up_axis": "+Z",
        "linear_unit": "metre",
        "global_orient_application": "smplh_lbs",
        "root_translation_frame": "amass_source_frame",
        "court_orientation": "configured_positive_z_yaw_only",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "plcs_amass_smplh_y_up_v1"),
        ("up_axis", "+Y"),
        ("global_orient_application", "court_placement"),
        ("root_translation_frame", "axis_converted"),
        ("court_orientation", "x_quarter_turn_then_yaw"),
    ),
)
def test_coordinate_contract_rejects_alternate_coordinate_meaning(
    field: str,
    value: str,
) -> None:
    payload = PLCS_COORDINATE_CONTRACT.to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match="does not match the v5 schema"):
        PLCSCoordinateContract.from_dict(payload)


def test_coordinate_contract_rejects_missing_or_additional_fields() -> None:
    missing = PLCS_COORDINATE_CONTRACT.to_dict()
    del missing["root_translation_frame"]
    additional = PLCS_COORDINATE_CONTRACT.to_dict()
    additional["legacy_axis_conversion"] = True

    with pytest.raises(ValueError, match="does not match the v5 schema"):
        PLCSCoordinateContract.from_dict(missing)
    with pytest.raises(ValueError, match="does not match the v5 schema"):
        PLCSCoordinateContract.from_dict(additional)


def test_support_plane_round_trips_exact_full_surface_frame_zero_evidence() -> None:
    support = PLCSSourceSupportPlane.from_surface_minimum(
        initial_root_translation_z_m=0.72,
        support_local_z_m=-0.93,
    )

    payload = support.to_dict()
    parsed = PLCSSourceSupportPlane.from_dict(payload)

    assert parsed == support
    assert payload == {
        "schema": PLCS_SUPPORT_PLANE_SCHEMA,
        "source_frame_index": 0,
        "surface_definition": (
            "frame-0 posed full SMPL-H surface after pose blend, LBS, and "
            "global_orient; before root translation; minimum local Z"
        ),
        "vertex_count": SMPLH_SURFACE_VERTEX_COUNT,
        "initial_root_translation_z_m": 0.72,
        "support_local_z_m": -0.93,
        "support_plane_source_z_m": pytest.approx(-0.21),
        "placement_tolerance_m": PLCS_SUPPORT_PLACEMENT_TOLERANCE_M,
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema", "plcs_initial_foot_joint_support_v1", "schema"),
        ("source_frame_index", 1, "source frame 0"),
        ("vertex_count", 24, "6,890"),
        ("support_plane_source_z_m", 99.0, "initial trans.z plus local min Z"),
        ("placement_tolerance_m", 1.0e-3, "exactly 1e-5"),
    ),
)
def test_support_plane_rejects_inexact_provenance(
    field: str,
    value: object,
    error: str,
) -> None:
    payload = PLCSSourceSupportPlane.from_surface_minimum(
        initial_root_translation_z_m=0.72,
        support_local_z_m=-0.93,
    ).to_dict()
    mutated = deepcopy(payload)
    mutated[field] = value

    with pytest.raises(ValueError, match=error):
        PLCSSourceSupportPlane.from_dict(mutated)


def test_support_plane_rejects_unknown_fields_and_boolean_numbers() -> None:
    payload = PLCSSourceSupportPlane.from_surface_minimum(
        initial_root_translation_z_m=0.0,
        support_local_z_m=-1.0,
    ).to_dict()
    payload["legacy_ground_offset"] = 0.0

    with pytest.raises(ValueError, match="keys differ"):
        PLCSSourceSupportPlane.from_dict(payload)

    payload = PLCSSourceSupportPlane.from_surface_minimum(
        initial_root_translation_z_m=0.0,
        support_local_z_m=-1.0,
    ).to_dict()
    payload["initial_root_translation_z_m"] = False
    with pytest.raises(TypeError, match="must be numeric"):
        PLCSSourceSupportPlane.from_dict(payload)
