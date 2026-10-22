"""Generation-time target support and clipped ROI diagnostics, before JPEG."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

from src.synthetic_data_generation.dataset.court.components.labels import (
    attach_renderer_visibility_from_validated_arrays,
    project_court_semantics_v3,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera

from .core import QualityPolicy


def roi_mask(record: dict[str, Any], near_plane_m: float) -> np.ndarray:
    """Clip the target physical corner polygon in camera space, then image space."""
    camera = SceneCamera.from_dict(record["camera"])
    target = record["target_court"]["binding"]["court_instance_id"]
    court = next(
        c for c in record["projection"]["courts"] if c["court_instance_id"] == target
    )
    points = {
        p["physical_index"]: p["scene_xyz_m"]
        for cls in court["classes"]
        for p in cls["points"]
    }
    polygon = camera.camera_to_scene.inverse().apply(
        np.asarray([points[i] for i in (0, 1, 3, 2)])
    )
    clipped = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if a[2] >= near_plane_m:
            clipped.append(a)
        if (a[2] >= near_plane_m) != (b[2] >= near_plane_m):
            clipped.append(a + (b - a) * (near_plane_m - a[2]) / (b[2] - a[2]))
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    if len(clipped) >= 3:
        homogeneous = (
            np.asarray(clipped) @ np.asarray(camera.intrinsics).reshape(3, 3).T
        )
        pixels = np.ascontiguousarray(
            homogeneous[:, :2] / homogeneous[:, 2:], dtype=np.float32
        )
        frame = np.array(
            [
                [0, 0],
                [camera.width - 1, 0],
                [camera.width - 1, camera.height - 1],
                [0, camera.height - 1],
            ],
            dtype=np.float32,
        )
        area, clipped_pixels = cv2.intersectConvexConvex(pixels, frame)
        if area > 0 and clipped_pixels is not None:
            cv2.fillConvexPoly(mask, np.rint(clipped_pixels).astype(np.int32), 1)
    return cast(np.ndarray, mask.astype(bool))


def evaluate_render(
    record: dict[str, Any],
    arrays: NHTRenderArrays,
    *,
    layout: MultiCourtLayout,
    policy: QualityPolicy,
    near_plane_m: float,
) -> dict[str, Any]:
    camera = SceneCamera.from_dict(record["camera"])
    if (arrays.width, arrays.height) != (camera.width, camera.height):
        raise ValueError("Renderer dimensions disagree with planned camera.")
    projection = project_court_semantics_v3(camera, layout, near_plane_m=near_plane_m)
    visible = attach_renderer_visibility_from_validated_arrays(
        projection, alpha=arrays.alpha, depth=arrays.depth
    )
    updated = {**record, "projection": visible.to_dict()}
    target_id = record["target_court"]["binding"]["court_instance_id"]
    target = next(c for c in visible.courts if c.court_instance_id == target_id)
    target_visible = sum(
        p.renderer_visible is True for cls in target.classes for p in cls.points
    )
    roi = roi_mask(updated, near_plane_m)
    alpha, depth = arrays.alpha[..., 0], arrays.depth[..., 0]
    roi_holes = (
        float(((alpha < 0.01) | (depth <= 0))[roi].mean()) if roi.any() else None
    )
    diagnostics = {
        "schema": "court_scale_render_diagnostics_v1",
        "target_geometry_kp": target.in_frame_point_count,
        "target_visible_kp": target_visible,
        "all_visible_kp": visible.visible_point_count,
        "target_coverage": target.coverage_mode,
        "roi_pixels": int(roi.sum()),
        "roi_hole_fraction": roi_holes,
        "roi_alpha_lt_0_5": float((alpha[roi] < 0.5).mean()) if roi.any() else None,
        "roi_alpha_lt_0_9": float((alpha[roi] < 0.9).mean()) if roi.any() else None,
        "roi_alpha_p05": float(np.quantile(alpha[roi], 0.05)) if roi.any() else None,
        "positive_depth_fraction": float((depth > 0).mean()),
        "sensitivity_target_kp_ge_4": target_visible >= 4,
        "sensitivity_roi_holes_le_0_01": roi_holes <= 0.01
        if roi_holes is not None
        else None,
    }
    return {
        **updated,
        "qa": diagnostics,
        "reasons": [
            *record["reasons"],
            *quality_rejection_reasons(diagnostics, policy),
        ],
    }


def quality_rejection_reasons(qa: dict[str, Any], policy: QualityPolicy) -> list[str]:
    """The same explicit predicate governs generation and persisted QA validation."""
    reasons = []
    if qa["all_visible_kp"] == 0:
        reasons.append("no_renderer_visible_semantic_point")
    if (
        policy.minimum_target_visible_kp is not None
        and qa["target_visible_kp"] < policy.minimum_target_visible_kp
    ):
        reasons.append("target_visible_kp_below_explicit_threshold")
    if policy.maximum_roi_hole_fraction is not None and (
        qa["roi_hole_fraction"] is None
        or qa["roi_hole_fraction"] > policy.maximum_roi_hole_fraction
    ):
        reasons.append("roi_holes_above_explicit_threshold_or_no_roi")
    return reasons
