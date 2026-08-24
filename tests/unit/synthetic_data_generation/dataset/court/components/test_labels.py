from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_view import (
    camera_view_canonicalization,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_CLASS_NAMES_V2,
    AmbiguousCameraRelativeNearFarError,
    CourtProjectionV3,
    attach_renderer_visibility,
    camera_relative_physical_indices,
    project_court_semantics,
    project_court_semantics_for_version,
    project_court_semantics_v2,
    project_court_semantics_v3,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    COURT_KP_NAMES,
    OPPOSITE_COURT_END_INDEX,
)

_LEFT_RIGHT_PAIRS = ((0, 1), (2, 3), (4, 6), (5, 7), (8, 9), (10, 11))


def _camera() -> SceneCamera:
    center = np.asarray((0.0, -30.0, 12.0), dtype=np.float64)
    target: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center
    return SceneCamera(
        camera_id="render-camera",
        source_frame_index=0,
        width=640,
        height=480,
        intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="generated/render-camera.png",
    )


def _camera_at_y(y: float) -> SceneCamera:
    camera = _camera()
    matrix = camera.camera_to_scene.matrix()
    matrix[:3, 3] = (0.0, y, 12.0)
    return SceneCamera(
        camera_id=f"camera-y-{y}",
        source_frame_index=0,
        width=camera.width,
        height=camera.height,
        intrinsics=camera.intrinsics,
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path=f"generated/camera-y-{y}.png",
    )


def _look_at_camera(center: tuple[float, float, float]) -> SceneCamera:
    center_array = np.asarray(center, dtype=np.float64)
    target: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    forward = target - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_array
    return SceneCamera(
        camera_id=f"look-at-{center[0]:g}-{center[1]:g}",
        source_frame_index=0,
        width=640,
        height=480,
        intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="generated/look-at.png",
    )


def _identity_court() -> CourtInstance:
    transform = RigidTransform.identity()
    return CourtInstance(
        court_instance_id="court-identity",
        candidate_id="candidate-identity",
        scene_from_court=transform,
        court_from_scene=transform,
        fit_status="accepted",
        fit_metrics={"rms_error_m": 0.01},
        holdout_status="accepted",
        holdout_metrics={"rms_error_m": 0.02},
    )


def test_renderer_visibility_exposes_all_seven_classes(
    multi_court_layout: MultiCourtLayout,
) -> None:
    projection = project_court_semantics(_camera(), multi_court_layout)
    alpha: NDArray[np.float32] = np.zeros((480, 640, 1), dtype=np.float32)
    depth: NDArray[np.float32] = np.zeros((480, 640, 1), dtype=np.float32)
    for court in projection.courts:
        for semantic_class in court.classes:
            for point in semantic_class.points:
                if point.in_frame:
                    x = int(round(point.uv[0]))
                    y = int(round(point.uv[1]))
                    alpha[y, x, 0] = 1.0
                    depth[y, x, 0] = point.camera_depth_m
    visible = attach_renderer_visibility(
        projection,
        alpha=alpha,
        depth=depth,
        sample_radius_px=0,
    )
    assert visible.visible_class_names == SEMANTIC_CLASS_NAMES
    assert visible.visible_point_count > 0
    assert all(
        court.coverage_mode in {"full", "near_full", "partial"}
        for court in visible.courts
    )


def test_renderer_visibility_requires_alpha_and_positive_depth(
    multi_court_layout: MultiCourtLayout,
) -> None:
    projection = project_court_semantics(_camera(), multi_court_layout)
    alpha: NDArray[np.float32] = np.ones((480, 640, 1), dtype=np.float32)
    depth: NDArray[np.float32] = np.ones((480, 640, 1), dtype=np.float32)

    supported = attach_renderer_visibility(projection, alpha=alpha, depth=depth)
    without_alpha = attach_renderer_visibility(
        projection,
        alpha=np.zeros_like(alpha),
        depth=depth,
    )
    without_depth = attach_renderer_visibility(
        projection,
        alpha=alpha,
        depth=np.zeros_like(depth),
    )

    assert supported.visible_point_count > 0
    assert without_alpha.visible_point_count == 0
    assert without_depth.visible_point_count == 0


def test_v1_projection_remains_seven_classes_with_two_physical_points() -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id=court.court_instance_id,
    )
    projection = project_court_semantics_for_version(
        _camera_at_y(-30.0),
        layout,
        schema_version=CourtDatasetSchemaVersion.V1,
    )

    assert tuple(value.class_name for value in projection.courts[0].classes) == (
        SEMANTIC_CLASS_NAMES
    )
    assert all(len(value.points) == 2 for value in projection.courts[0].classes)
    assert sorted(
        point.physical_index
        for value in projection.courts[0].classes
        for point in value.points
    ) == list(range(14))


def test_v2_projection_is_fourteen_singletons_and_swaps_across_mid_plane() -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id=court.court_instance_id,
    )
    negative = project_court_semantics_v2(_camera_at_y(-30.0), layout)
    positive = project_court_semantics_v2(_camera_at_y(30.0), layout)

    for projection, expected_indices in (
        (negative, tuple(range(14))),
        (positive, OPPOSITE_COURT_END_INDEX),
    ):
        classes = projection.courts[0].classes
        assert tuple(value.class_id for value in classes) == tuple(range(14))
        assert tuple(value.class_name for value in classes) == COURT_KP_NAMES[:14]
        assert tuple(value.class_name for value in classes) == SEMANTIC_CLASS_NAMES_V2
        assert all(len(value.points) == 1 for value in classes)
        assert tuple(value.points[0].physical_index for value in classes) == (
            expected_indices
        )
        assert set(value.points[0].physical_index for value in classes) == set(
            range(14)
        )


@pytest.mark.parametrize("y", [-1.0e-6, 0.0, 1.0e-6])
def test_v2_mid_plane_ambiguity_has_court_qualified_rejection_reason(y: float) -> None:
    court = _identity_court()
    with pytest.raises(
        AmbiguousCameraRelativeNearFarError,
        match="ambiguous_camera_relative_near_far:court-identity",
    ) as error:
        camera_relative_physical_indices(_camera_at_y(y), court)

    assert error.value.court_instance_id == "court-identity"
    assert error.value.reason == ("ambiguous_camera_relative_near_far:court-identity")


def test_v2_renderer_visibility_uses_ordered_fourteen_class_inventory() -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id=court.court_instance_id,
    )
    projection = project_court_semantics_v2(_camera_at_y(-30.0), layout)
    alpha: NDArray[np.float32] = np.ones((480, 640, 1), dtype=np.float32)
    depth: NDArray[np.float32] = np.ones((480, 640, 1), dtype=np.float32)

    visible = attach_renderer_visibility(projection, alpha=alpha, depth=depth)

    assert visible.visible_class_names == SEMANTIC_CLASS_NAMES_V2
    assert visible.visible_point_count == 14


@pytest.mark.parametrize(
    ("center", "expected_indices"),
    [
        ((0.0, -30.0, 12.0), tuple(range(14))),
        ((0.0, 30.0, 12.0), CAMERA_VIEW_HALF_TURN_INDEX),
        ((-6.0, -30.0, 12.0), tuple(range(14))),
        ((6.0, -30.0, 12.0), tuple(range(14))),
        ((-6.0, 30.0, 12.0), CAMERA_VIEW_HALF_TURN_INDEX),
        ((6.0, 30.0, 12.0), CAMERA_VIEW_HALF_TURN_INDEX),
        ((30.0, -12.5, 12.0), tuple(range(14))),
        ((-30.0, 12.5, 12.0), CAMERA_VIEW_HALF_TURN_INDEX),
    ],
)
def test_v3_baseline_exterior_projection_orders_all_left_right_pairs(
    center: tuple[float, float, float],
    expected_indices: tuple[int, ...],
) -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-40.0, -40.0, -1.0, 40.0, 40.0, 20.0),
        primary_court_instance_id=court.court_instance_id,
    )
    projection = project_court_semantics_v3(_look_at_camera(center), layout)
    classes = projection.courts[0].classes

    assert tuple(value.points[0].physical_index for value in classes) == expected_indices
    assert set(expected_indices) == set(range(14))
    for left, right in _LEFT_RIGHT_PAIRS:
        assert classes[left].points[0].uv[0] < classes[right].points[0].uv[0]
    center_array = np.asarray(center, dtype=np.float64)
    for far_indices, near_indices in (
        ((0, 1), (2, 3)),
        ((4, 6), (5, 7)),
        ((8, 9), (10, 11)),
    ):
        far_distance = np.mean(
            [
                np.linalg.norm(
                    np.asarray(classes[index].points[0].scene_xyz_m) - center_array
                )
                for index in far_indices
            ]
        )
        near_distance = np.mean(
            [
                np.linalg.norm(
                    np.asarray(classes[index].points[0].scene_xyz_m) - center_array
                )
                for index in near_indices
            ]
        )
        assert near_distance < far_distance


def test_v3_renderer_visibility_preserves_full_half_turn_physical_provenance() -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-40.0, -40.0, -1.0, 40.0, 40.0, 20.0),
        primary_court_instance_id=court.court_instance_id,
    )
    projection = project_court_semantics_v3(
        _look_at_camera((0.0, 30.0, 12.0)),
        layout,
    )
    before = projection.courts[0]
    visible = attach_renderer_visibility(
        projection,
        alpha=np.ones((480, 640, 1), dtype=np.float32),
        depth=np.ones((480, 640, 1), dtype=np.float32),
    )
    after = visible.courts[0]

    assert tuple(
        value.points[0].physical_index for value in after.classes
    ) == CAMERA_VIEW_HALF_TURN_INDEX
    assert tuple(
        value.points[0].scene_xyz_m for value in after.classes
    ) == tuple(value.points[0].scene_xyz_m for value in before.classes)
    assert tuple(
        value.points[0].in_front for value in after.classes
    ) == tuple(value.points[0].in_front for value in before.classes)
    assert tuple(
        value.points[0].in_frame for value in after.classes
    ) == tuple(value.points[0].in_frame for value in before.classes)
    assert tuple(
        value.points[0].renderer_visible for value in after.classes
    ) == tuple(value.points[0].in_frame for value in before.classes)


def test_v3_projection_rejects_duplicate_or_noncanonical_physical_inventory() -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-40.0, -40.0, -1.0, 40.0, 40.0, 20.0),
        primary_court_instance_id=court.court_instance_id,
    )
    classes = list(
        project_court_semantics_v3(
            _look_at_camera((0.0, 30.0, 12.0)),
            layout,
        ).courts[0].classes
    )
    duplicate_point = replace(
        classes[1].points[0],
        physical_index=classes[0].points[0].physical_index,
    )
    classes[1] = replace(classes[1], points=(duplicate_point,))

    with pytest.raises(ValueError, match="preserve each physical index"):
        CourtProjectionV3(
            court_instance_id=court.court_instance_id,
            classes=tuple(classes),
        )


def test_v3_multi_court_side_decisions_are_independent() -> None:
    identity = _identity_court()
    half_turn_matrix = np.eye(4, dtype=np.float64)
    half_turn_matrix[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    half_turn = RigidTransform.from_matrix(half_turn_matrix)
    rotated = CourtInstance(
        court_instance_id="court-rotated",
        candidate_id="candidate-rotated",
        scene_from_court=half_turn,
        court_from_scene=half_turn.inverse(),
        fit_status="accepted",
        fit_metrics={"rms_error_m": 0.01},
        holdout_status="accepted",
        holdout_metrics={"rms_error_m": 0.02},
    )
    layout = MultiCourtLayout(
        courts=(identity, rotated),
        complex_bounds_scene=(-40.0, -40.0, -1.0, 40.0, 40.0, 20.0),
        primary_court_instance_id=identity.court_instance_id,
    )

    projection = project_court_semantics_v3(
        _look_at_camera((-30.0, -2.0, 12.0)), layout
    )
    orders = tuple(
        tuple(value.points[0].physical_index for value in court.classes)
        for court in projection.courts
    )
    assert orders == (tuple(range(14)), CAMERA_VIEW_HALF_TURN_INDEX)
    for class_id in range(14):
        assert projection.courts[0].classes[class_id].points[0].uv == pytest.approx(
            projection.courts[1].classes[class_id].points[0].uv
        )


@pytest.mark.parametrize(
    ("center", "expected_indices"),
    [
        ((-30.0, -2.0, 12.0), tuple(range(14))),
        ((-30.0, 2.0, 12.0), CAMERA_VIEW_HALF_TURN_INDEX),
    ],
)
def test_v3_lateral_projection_accepts_reversed_u_order_and_round_trips(
    center: tuple[float, float, float],
    expected_indices: tuple[int, ...],
) -> None:
    court = _identity_court()
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-40.0, -40.0, -1.0, 40.0, 40.0, 20.0),
        primary_court_instance_id=court.court_instance_id,
    )
    camera = _look_at_camera(center)
    projection = project_court_semantics_v3(camera, layout)
    classes = projection.courts[0].classes

    assert tuple(value.points[0].physical_index for value in classes) == expected_indices
    pair_order = tuple(
        classes[left].points[0].uv[0] < classes[right].points[0].uv[0]
        for left, right in _LEFT_RIGHT_PAIRS
    )
    assert any(pair_order)
    assert not all(pair_order)

    center_array = np.asarray(center, dtype=np.float64)
    for far_indices, near_indices in (
        ((0, 1), (2, 3)),
        ((4, 6), (5, 7)),
        ((8, 9), (10, 11)),
    ):
        far_distance = np.mean(
            [
                np.linalg.norm(
                    np.asarray(classes[index].points[0].scene_xyz_m) - center_array
                )
                for index in far_indices
            ]
        )
        near_distance = np.mean(
            [
                np.linalg.norm(
                    np.asarray(classes[index].points[0].scene_xyz_m) - center_array
                )
                for index in near_indices
            ]
        )
        assert near_distance < far_distance

    canonical = camera_view_canonicalization(camera, court)
    known_court = np.asarray(((-5.485, 11.885, 0.0),), dtype=np.float64)
    original_camera = camera.camera_to_scene.inverse().apply(known_court)
    canonical_camera = canonical.camera_from_canonical.apply(
        canonical.canonical_from_court.apply(known_court)
    )
    np.testing.assert_allclose(original_camera, canonical_camera, atol=1.0e-12)
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    original_homogeneous = original_camera @ intrinsics.T
    canonical_homogeneous = canonical_camera @ intrinsics.T
    np.testing.assert_allclose(
        original_homogeneous[:, :2] / original_homogeneous[:, 2:3],
        canonical_homogeneous[:, :2] / canonical_homogeneous[:, 2:3],
        atol=1.0e-12,
    )
