"""Pure shared CourtKP20 camera-view and reference-frame contract tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID,
    CAMERA_VIEW_V2_SELECTOR,
    COURT_KEYPOINT_METADATA_KEY,
    COURT_VIEW_METADATA_KEY,
    IDENTITY_COURT_KP20_INDEX,
    IDENTITY_ROTATION_3D,
    PHYSICAL_COURT_TARGET_FRAME_ID,
    PHYSICAL_COURTKP20_CONTRACT_ID,
    PHYSICAL_V1_SELECTOR,
    REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID,
    RZ_PI_ROTATION_3D,
    CameraCourtViewError,
    CourtKeypointArtifactMetadata,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtKeypointMappingError,
    CourtReferenceFrameError,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    InvalidCourtKeypointMetadataError,
    MissingCourtKeypointMetadataError,
    MixedCourtKeypointMetadataError,
    align_court_keypoints_to_reference,
    apply_court_view_record,
    build_court_view_record,
    build_physical_court_provenance,
    build_reference_frame_provenance,
    camera_extrinsics_physical_to_target,
    camera_extrinsics_target_to_physical,
    classify_camera_court_side,
    court_headings_physical_to_target,
    court_headings_target_to_physical,
    court_points_physical_to_target,
    court_points_target_to_physical,
    court_vectors_physical_to_target,
    court_world_joints_physical_to_target,
    inject_court_keypoint_artifact_metadata,
    inject_scene_court_keypoint_metadata,
    reference_court_keypoint_indices,
    resolve_court_keypoint_contract,
    resolve_reference_camera_local_index,
    validate_court_keypoint_mapping,
    validate_dataset_court_keypoint_contract_documents,
    validate_reference_frame_provenance,
)
from src.utils.schema.court import COURT_KP20_HALF_TURN_INDEX

_DATASET_SCHEMA_ID = "test_generated_dataset_v2"


def _contract(selector: str = CAMERA_VIEW_V2_SELECTOR) -> CourtKeypointContract:
    return resolve_court_keypoint_contract(selector)


def _negative(camera_id: str = "cam_neg") -> CourtViewRecord:
    return build_court_view_record(
        camera_id=camera_id,
        camera_center_court_m=(7.0, -14.0, 6.0),
        contract=_contract(),
    )


def _positive(camera_id: str = "cam_pos") -> CourtViewRecord:
    return build_court_view_record(
        camera_id=camera_id,
        camera_center_court_m=(-3.0, 15.0, 8.0),
        contract=_contract(),
    )


def _artifact(
    selector: str = CAMERA_VIEW_V2_SELECTOR,
) -> CourtKeypointArtifactMetadata:
    return CourtKeypointArtifactMetadata.from_contract(
        _contract(selector),
        dataset_schema_id=_DATASET_SCHEMA_ID,
    )


def _root_document(selector: str = CAMERA_VIEW_V2_SELECTOR) -> dict[str, object]:
    document: dict[str, object] = inject_court_keypoint_artifact_metadata(
        {"root": True},
        _artifact(selector),
        location="dataset/meta.json",
    )
    return document


def _scene_document(selector: str = CAMERA_VIEW_V2_SELECTOR) -> dict[str, object]:
    contract = _contract(selector)
    if selector == PHYSICAL_V1_SELECTOR:
        records = (
            build_court_view_record(
                camera_id="cam_neg",
                camera_center_court_m=(7.0, -14.0, 6.0),
                contract=contract,
            ),
            build_court_view_record(
                camera_id="cam_pos",
                camera_center_court_m=(-3.0, 15.0, 8.0),
                contract=contract,
            ),
        )
    else:
        records = (_negative(), _positive())
    document: dict[str, object] = inject_scene_court_keypoint_metadata(
        {"scene": "scene_a"},
        _artifact(selector),
        records,
        location="dataset/scenes/scene_a/meta.json",
    )
    return document


def test_public_selectors_resolve_exact_ids() -> None:
    physical = _contract(PHYSICAL_V1_SELECTOR)
    camera_view = _contract(CAMERA_VIEW_V2_SELECTOR)

    assert (
        physical.contract_id,
        physical.target_frame_id,
        physical.camera_view_semantics,
    ) == (
        PHYSICAL_COURTKP20_CONTRACT_ID,
        PHYSICAL_COURT_TARGET_FRAME_ID,
        False,
    )
    assert (
        camera_view.contract_id,
        camera_view.target_frame_id,
        camera_view.camera_view_semantics,
    ) == (
        CAMERA_VIEW_COURTKP20_RZPI_CONTRACT_ID,
        REFERENCE_CAMERA_COURT_RZPI_TARGET_FRAME_ID,
        True,
    )
    with pytest.raises(ValueError, match="Unknown court keypoint selector"):
        _contract("v2")


def test_camera_view_builder_uses_identity_or_exact_rzpi_full20() -> None:
    negative = _negative()
    positive = _positive()

    assert negative.semantic_to_physical == IDENTITY_COURT_KP20_INDEX
    assert negative.canonical_from_physical == IDENTITY_ROTATION_3D
    assert positive.semantic_to_physical == COURT_KP20_HALF_TURN_INDEX
    assert positive.canonical_from_physical == RZ_PI_ROTATION_3D
    assert negative.contract_id == positive.contract_id


@pytest.mark.parametrize("y", [-1e-6, 0.0, 1e-6])
def test_inclusive_camera_mid_plane_is_rejected(y: float) -> None:
    with pytest.raises(CameraCourtViewError, match=r"abs\(C_y\).+<="):
        build_court_view_record(
            camera_id="boundary",
            camera_center_court_m=(9.0, y, 7.0),
            contract=_contract(),
        )


@pytest.mark.parametrize(
    "center",
    [
        (np.nan, -1.0, 2.0),
        (0.0, np.inf, 2.0),
        (0.0, -1.0, -np.inf),
    ],
)
def test_nonfinite_camera_center_is_rejected(center: tuple[float, ...]) -> None:
    with pytest.raises(CameraCourtViewError, match="finite"):
        classify_camera_court_side(center)


def test_physical_v1_keeps_explicit_identity_at_mid_plane() -> None:
    record = build_court_view_record(
        camera_id="legacy_mid",
        camera_center_court_m=(1.0, 0.0, 4.0),
        contract=_contract(PHYSICAL_V1_SELECTOR),
    )
    assert record.semantic_to_physical == IDENTITY_COURT_KP20_INDEX
    assert record.canonical_from_physical == IDENTITY_ROTATION_3D


@pytest.mark.parametrize(
    "mapping",
    [
        tuple(range(19)),
        tuple(range(19)) + (18,),
        np.arange(20).reshape(20, 1),
        tuple(range(1, 20)) + (0,),
        tuple(float(index) for index in range(20)),
    ],
)
def test_mapping_shape_bijection_and_involution_are_strict(mapping: object) -> None:
    with pytest.raises(CourtKeypointMappingError):
        validate_court_keypoint_mapping(mapping)


def test_view_record_round_trip_revalidates_center_mapping_and_matrix() -> None:
    record = _positive()
    assert type(record).from_mapping(record.to_dict(), location="camera") == record

    for field, value in (
        ("semantic_to_physical", list(range(20))),
        ("canonical_from_physical", [list(row) for row in IDENTITY_ROTATION_3D]),
        ("camera_center_court_m", [0.0, 0.0, 3.0]),
        ("coordinate_frame", "reference_court"),
    ):
        mutated = record.to_dict()
        mutated[field] = value
        with pytest.raises(InvalidCourtKeypointMetadataError, match=field):
            type(record).from_mapping(mutated, location="camera")


def test_physical_projection_is_reordered_once_for_uv_and_visibility() -> None:
    physical_uv: np.ndarray = np.arange(40, dtype=np.float64).reshape(20, 2)
    physical_visibility: np.ndarray = np.arange(20, dtype=np.float64)
    view = _positive()

    disk_uv = apply_court_view_record(physical_uv, view, keypoint_axis=0)
    disk_visibility = apply_court_view_record(
        physical_visibility,
        view,
        keypoint_axis=0,
    )

    np.testing.assert_array_equal(
        disk_uv,
        physical_uv[np.asarray(COURT_KP20_HALF_TURN_INDEX)],
    )
    np.testing.assert_array_equal(
        disk_visibility,
        physical_visibility[np.asarray(COURT_KP20_HALF_TURN_INDEX)],
    )


def test_reference_alignment_implements_h_source_inverse_after_h_reference() -> None:
    negative = _negative()
    positive = _positive()
    physical: np.ndarray = np.arange(40, dtype=np.float64).reshape(20, 2)
    negative_disk = apply_court_view_record(physical, negative, keypoint_axis=0)
    positive_disk = apply_court_view_record(physical, positive, keypoint_axis=0)

    remap = reference_court_keypoint_indices(negative, positive)
    assert remap == COURT_KP20_HALF_TURN_INDEX
    np.testing.assert_array_equal(
        align_court_keypoints_to_reference(
            negative_disk,
            negative,
            positive,
            keypoint_axis=0,
        ),
        positive_disk,
    )
    np.testing.assert_array_equal(
        align_court_keypoints_to_reference(
            positive_disk,
            positive,
            positive,
            keypoint_axis=0,
        ),
        positive_disk,
    )


def test_time_axis_court_alignment_precedes_tracking_prefix() -> None:
    negative = _negative()
    positive = _positive()
    physical: np.ndarray = np.arange(2 * 20 * 2, dtype=np.float32).reshape(2, 20, 2)
    negative_disk = apply_court_view_record(physical, negative, keypoint_axis=1)

    aligned = align_court_keypoints_to_reference(
        negative_disk,
        negative,
        positive,
        keypoint_axis=1,
    )

    expected = physical[:, np.asarray(COURT_KP20_HALF_TURN_INDEX)]
    np.testing.assert_array_equal(aligned, expected)
    np.testing.assert_array_equal(aligned[:, :14], expected[:, :14])


def test_stable_reference_identity_resolves_after_camera_reorder() -> None:
    selected = (_positive("cam_9"), _negative("cam_2"))

    assert resolve_reference_camera_local_index(["cam_9", "cam_2"], "cam_2") == 1
    provenance = build_reference_frame_provenance(
        selected,
        reference_camera_id="cam_2",
    )
    assert provenance.reference_camera_id == "cam_2"
    assert provenance.reference_camera_local_index == 1
    assert validate_reference_frame_provenance(provenance, selected) == selected[1]

    with pytest.raises(CourtReferenceFrameError, match="not in selected"):
        resolve_reference_camera_local_index(["cam_9", "cam_2"], "cam_0")
    with pytest.raises(CourtReferenceFrameError, match="unique"):
        resolve_reference_camera_local_index(["cam_2", "cam_2"], "cam_2")


def test_root_scene_camera_metadata_round_trip_is_exact() -> None:
    result = validate_dataset_court_keypoint_contract_documents(
        root_metadata=_root_document(),
        scene_metadata={"scene_a": _scene_document()},
        runtime_contract=_contract(),
        expected_dataset_schema_id=_DATASET_SCHEMA_ID,
    )

    assert result.contract == _contract()
    assert result.metadata == _artifact()
    assert result.legacy_metadata_free is False
    assert [view.camera_id for view in result.scenes[0].court_views] == [
        "cam_neg",
        "cam_pos",
    ]


@pytest.mark.parametrize(
    ("level", "field", "value"),
    [
        ("root", "dataset_schema_id", "plcs_generated_dataset_v2"),
        ("scene", "coordinate_frame", "reference_court"),
        ("camera", "contract_id", PHYSICAL_COURTKP20_CONTRACT_ID),
        ("camera", "camera_center_court_m", [0.0, 0.0, 2.0]),
    ],
)
def test_root_scene_camera_metadata_mutations_are_rejected(
    level: str,
    field: str,
    value: object,
) -> None:
    root = _root_document()
    scene = _scene_document()
    if level == "root":
        raw = deepcopy(root[COURT_KEYPOINT_METADATA_KEY])
        assert isinstance(raw, dict)
        raw[field] = value
        root[COURT_KEYPOINT_METADATA_KEY] = raw
    elif level == "scene":
        raw = deepcopy(scene[COURT_KEYPOINT_METADATA_KEY])
        assert isinstance(raw, dict)
        raw[field] = value
        scene[COURT_KEYPOINT_METADATA_KEY] = raw
    else:
        cameras = deepcopy(scene[COURT_VIEW_METADATA_KEY])
        assert isinstance(cameras, list) and isinstance(cameras[0], dict)
        cameras[0][field] = value
        scene[COURT_VIEW_METADATA_KEY] = cameras

    with pytest.raises(
        (InvalidCourtKeypointMetadataError, CourtKeypointContractMismatchError)
    ):
        validate_dataset_court_keypoint_contract_documents(
            root_metadata=root,
            scene_metadata={"scene_a": scene},
            runtime_contract=_contract(),
            expected_dataset_schema_id=_DATASET_SCHEMA_ID,
        )


@pytest.mark.parametrize(
    ("root", "scene"),
    [
        ({}, _scene_document()),
        (_root_document(), {}),
        (_root_document(), {COURT_KEYPOINT_METADATA_KEY: _artifact().to_dict()}),
    ],
)
def test_mixed_root_scene_camera_metadata_is_rejected(
    root: dict[str, object],
    scene: dict[str, object],
) -> None:
    with pytest.raises(MixedCourtKeypointMetadataError):
        validate_dataset_court_keypoint_contract_documents(
            root_metadata=root,
            scene_metadata={"scene_a": scene},
            runtime_contract=_contract(),
            expected_dataset_schema_id=_DATASET_SCHEMA_ID,
        )


def test_metadata_free_dataset_requires_explicit_physical_v1() -> None:
    legacy = validate_dataset_court_keypoint_contract_documents(
        root_metadata={},
        scene_metadata={"scene_a": {}},
        runtime_contract=_contract(PHYSICAL_V1_SELECTOR),
        expected_dataset_schema_id="blcs_generated_dataset_v1",
    )
    assert legacy.legacy_metadata_free is True

    with pytest.raises(MissingCourtKeypointMetadataError, match="physical_v1"):
        validate_dataset_court_keypoint_contract_documents(
            root_metadata={},
            scene_metadata={"scene_a": {}},
            runtime_contract=_contract(CAMERA_VIEW_V2_SELECTOR),
            expected_dataset_schema_id=_DATASET_SCHEMA_ID,
        )


def test_float64_point_vector_heading_and_world_joint_transform_round_trip() -> None:
    provenance = build_reference_frame_provenance(
        (_negative(), _positive()),
        reference_camera_id="cam_pos",
    )
    points = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]], dtype=np.float64)
    vectors = np.asarray([[3.0, -2.0, 1.0]], dtype=np.float64)
    headings = np.asarray([[0.6, 0.8], [-1.0, 0.0]], dtype=np.float64)
    joints = points.reshape(1, 2, 3)

    np.testing.assert_array_equal(
        court_points_physical_to_target(points, provenance),
        points * np.asarray([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        court_vectors_physical_to_target(vectors, provenance),
        vectors * np.asarray([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        court_headings_physical_to_target(headings, provenance),
        -headings,
    )
    np.testing.assert_array_equal(
        court_world_joints_physical_to_target(joints, provenance),
        joints * np.asarray([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        court_points_target_to_physical(
            court_points_physical_to_target(points, provenance),
            provenance,
        ),
        points,
    )
    np.testing.assert_array_equal(
        court_headings_target_to_physical(
            court_headings_physical_to_target(headings, provenance),
            provenance,
        ),
        headings,
    )


def test_float32_torch_transform_preserves_dtype_with_runtime_tolerance() -> None:
    provenance = build_reference_frame_provenance(
        (_positive(),),
        reference_camera_id="cam_pos",
    )
    points = torch.tensor([[1.25, -2.5, 0.75]], dtype=torch.float32)

    transformed = court_points_physical_to_target(points, provenance)

    assert transformed.dtype == torch.float32
    torch.testing.assert_close(
        transformed,
        torch.tensor([[-1.25, 2.5, 0.75]], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_extrinsic_transform_preserves_projection_exactly_in_float64() -> None:
    provenance = build_reference_frame_provenance(
        (_positive(),),
        reference_camera_id="cam_pos",
    )
    center_physical = np.asarray([7.0, -12.0, -5.0], dtype=np.float64)
    rotation_camera_from_physical = np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    point_physical = np.asarray([2.5, 3.0, 1.2], dtype=np.float64)
    point_target = court_points_physical_to_target(point_physical, provenance)
    center_target, rotation_camera_from_target = camera_extrinsics_physical_to_target(
        center_physical,
        rotation_camera_from_physical,
        provenance,
    )

    camera_physical = rotation_camera_from_physical @ (point_physical - center_physical)
    camera_target = rotation_camera_from_target @ (point_target - center_target)
    np.testing.assert_allclose(camera_target, camera_physical, rtol=0.0, atol=1e-14)

    focal_xy = np.asarray([917.0, 863.0], dtype=np.float64)
    principal_xy = np.asarray([641.5, 359.5], dtype=np.float64)

    def project(camera_point: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        depth = float(camera_point[2])
        uv: NDArray[np.float64] = focal_xy * camera_point[:2] / depth + principal_xy
        return uv, depth

    physical_uv, physical_depth = project(camera_physical)
    target_uv, target_depth = project(camera_target)
    assert physical_depth > 0.0
    np.testing.assert_allclose(target_uv, physical_uv, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(target_depth, physical_depth, rtol=0.0, atol=1e-14)

    restored_center, restored_rotation = camera_extrinsics_target_to_physical(
        center_target,
        rotation_camera_from_target,
        provenance,
    )
    np.testing.assert_array_equal(restored_center, center_physical)
    np.testing.assert_array_equal(restored_rotation, rotation_camera_from_physical)


def test_provenance_serialization_validates_inverse_and_selected_identity() -> None:
    selected = (_negative(), _positive())
    provenance = build_reference_frame_provenance(
        selected,
        reference_camera_id="cam_pos",
    )
    parsed = CourtReferenceFrameProvenance.from_mapping(
        provenance.to_dict(),
        location="prediction.frame",
    )
    assert parsed == provenance
    assert validate_reference_frame_provenance(parsed, selected) == selected[1]

    bad_inverse = provenance.to_dict()
    bad_inverse["physical_from_reference"] = [list(row) for row in IDENTITY_ROTATION_3D]
    with pytest.raises(InvalidCourtKeypointMetadataError, match="inverse"):
        CourtReferenceFrameProvenance.from_mapping(
            bad_inverse,
            location="prediction.frame",
        )

    bad_index = provenance.to_dict()
    bad_index["reference_camera_local_index"] = 0
    parsed_bad_index = CourtReferenceFrameProvenance.from_mapping(
        bad_index,
        location="prediction.frame",
    )
    with pytest.raises(CourtReferenceFrameError, match="local index"):
        validate_reference_frame_provenance(parsed_bad_index, selected)


def test_physical_provenance_is_identity_and_has_no_reference() -> None:
    provenance = build_physical_court_provenance()
    values = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)

    assert provenance.reference_camera_id is None
    assert provenance.reference_camera_local_index is None
    np.testing.assert_array_equal(
        court_points_physical_to_target(values, provenance),
        values,
    )


def test_provenance_device_transfer_is_an_identity_operation() -> None:
    provenance = build_reference_frame_provenance(
        (_negative(), _positive()),
        reference_camera_id="cam_pos",
    )
    fields_before = provenance.to_dict()

    transferred = provenance.to(
        torch.device("cpu"),
        dtype=torch.float64,
        non_blocking=True,
    )

    assert transferred is provenance
    assert transferred.to_dict() == fields_before
