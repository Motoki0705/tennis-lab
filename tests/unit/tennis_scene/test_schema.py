"""Canonical scene validity and CourtKP20 provenance contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
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
    SCENE_MASK_FIELDS,
    SCENE_REASON_FIELDS,
    SceneResult,
    attach_court_keypoint_provenance,
    validate_court_keypoint_provenance,
    validate_scene_result_arrays,
)


def test_v2_validator_accepts_origins_and_missing_points(v2_scene: SceneResult) -> None:
    # Frame 0 is a valid origin, including root, joints, yaw and ball. Frame 1
    # has the same zero geometry but is missing, despite accepted 2D observations.
    validate_scene_result_arrays(v2_scene)
    np.testing.assert_array_equal(v2_scene.player_valid, [[True, False, True]])
    np.testing.assert_array_equal(v2_scene.player_heading_valid, [[True, False, True]])
    np.testing.assert_array_equal(v2_scene.player_kp_3d_vis, np.repeat([[[True], [False], [True]]], 17, axis=-1))
    np.testing.assert_array_equal(v2_scene.ball_3d_valid, [True, False, False])


def test_v2_validator_allows_a_root_without_heading_or_mesh(v2_scene: SceneResult) -> None:
    assert v2_scene.player_heading_valid is not None
    v2_scene.player_heading_valid[0, 2] = False
    validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize(
    "field",
    [
        "player_position", "player_yaw", "player_kp_3d", "ball_3d",
        "smpl_body_pose", "smpl_global_orient", "smpl_vertices_local",
    ],
)
def test_v2_validator_rejects_nonzero_missing_geometry(v2_scene: SceneResult, field: str) -> None:
    value = getattr(v2_scene, field)
    value[1 if field == "ball_3d" else (0, 1)] = 1.0
    with pytest.raises(ValueError, match=f"{field} must be zero in invalid"):
        validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize("field", SCENE_REASON_FIELDS)
@pytest.mark.parametrize("valid_frame", [True, False], ids=["valid-with-rejection", "missing-with-zero-reason"])
def test_v2_validator_requires_zero_reason_exactly_for_valid_points(v2_scene: SceneResult, field: str, valid_frame: bool) -> None:
    codes = getattr(v2_scene, field)
    frame = 0 if valid_frame else 1
    codes[frame if field == "ball_rejection_code" else (0, frame)] = 3 if valid_frame else 0
    with pytest.raises(ValueError, match=f"{field}: zero reason must mean valid"):
        validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize("field", SCENE_MASK_FIELDS)
@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_v2_validator_rejects_mask_type_and_shape(v2_scene: SceneResult, field: str, fault: str) -> None:
    value = getattr(v2_scene, field)
    setattr(v2_scene, field, value.astype(np.uint8) if fault == "dtype" else value[..., None])
    with pytest.raises(ValueError, match=f"{field} must be boolean"):
        validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize("field", SCENE_REASON_FIELDS)
@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_v2_validator_rejects_reason_type_and_shape(v2_scene: SceneResult, field: str, fault: str) -> None:
    value = getattr(v2_scene, field)
    setattr(v2_scene, field, value.astype(np.int64) if fault == "dtype" else value[..., None])
    with pytest.raises(ValueError, match=f"{field} must be uint8 with the associated validity shape"):
        validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("player_position", "player position"), ("player_yaw", "player_yaw"),
        ("player_kp_3d", "player_kp_3d"), ("ball_3d", "ball_3d"),
        ("smpl_vertices_local", "SMPL vertices"),
    ],
)
@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_v2_validator_rejects_geometry_type_and_shape(v2_scene: SceneResult, field: str, message: str, fault: str) -> None:
    value = getattr(v2_scene, field)
    setattr(v2_scene, field, value.astype(np.float64) if fault == "dtype" else value[..., None])
    with pytest.raises(ValueError, match=message):
        validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s.player_heading_valid.__setitem__((0, 1), True), id="heading-requires-root"),
        pytest.param(lambda s: s.player_heading_valid.__setitem__((0, 0), False), id="mesh-requires-heading"),
        pytest.param(lambda s: s.player_smpl_valid.__setitem__((0, 1), True), id="mesh-requires-root"),
    ],
)
def test_v2_validator_rejects_heading_and_mesh_without_dependencies(v2_scene: SceneResult, mutate: Callable[[SceneResult], None]) -> None:
    mutate(v2_scene)
    with pytest.raises(ValueError, match="body/heading validity requires a valid player root"):
        validate_scene_result_arrays(v2_scene)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(lambda s: s.player_observed.__setitem__((0, 0), False), "lacks accepted observations", id="player-needs-observations"),
        pytest.param(lambda s: s.ball_vis.__setitem__((1, 0), False), "at least two views", id="ball-needs-two-views"),
        pytest.param(lambda s: s.metadata.pop("court_reference"), "resolved camera reference", id="geometry-needs-reference"),
    ],
)
def test_v2_validator_rejects_geometry_without_evidence(v2_scene: SceneResult, mutate: Callable[[SceneResult], None], message: str) -> None:
    mutate(v2_scene)
    with pytest.raises(ValueError, match=message):
        validate_scene_result_arrays(v2_scene)


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
