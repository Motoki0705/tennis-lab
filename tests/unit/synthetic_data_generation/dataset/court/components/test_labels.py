from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_CLASS_NAMES_V2,
    AmbiguousCameraRelativeNearFarError,
    attach_renderer_visibility,
    camera_relative_physical_indices,
    project_court_semantics,
    project_court_semantics_for_version,
    project_court_semantics_v2,
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
    COURT_KP_NAMES,
    OPPOSITE_COURT_END_INDEX,
)


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
