from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
    attach_renderer_visibility,
    project_court_semantics,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
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
    assert all(court.coverage_mode in {"full", "near_full", "partial"} for court in visible.courts)


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
