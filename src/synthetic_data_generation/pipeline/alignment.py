"""Deterministic geometric court-frame alignment from a standard NHT scene."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import AlignmentConfig
from .court_evidence import (
    evaluate,
    fit_court,
    observations,
    plane_frame,
    publish_evidence,
    rasterize,
    sparse_control_observations,
    split_cameras,
    transforms,
)
from .scene import StandardScene

COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77


def _percentile_extent(values: NDArray[np.float64]) -> float:
    return float(np.percentile(values, 99) - np.percentile(values, 1))


def _ground_plane(
    points: NDArray[np.float64], cameras: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, NDArray[np.bool_], float]:
    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered / len(centered)
    _, eigenvectors = np.linalg.eigh(covariance)
    candidate_normal = eigenvectors[:, 0]
    candidate_normal /= np.linalg.norm(candidate_normal)
    projected = points @ candidate_normal
    span = max(_percentile_extent(projected), 1.0e-9)
    tolerance = max(span * 0.025, np.linalg.norm(np.ptp(points, axis=0)) * 0.002)

    options = (
        (candidate_normal, float(np.percentile(projected, 5))),
        (-candidate_normal, float(np.percentile(-projected, 5))),
    )
    ranked = []
    for normal, offset in options:
        point_distance = points @ normal - offset
        camera_distance = cameras @ normal - offset
        support = np.abs(point_distance) <= tolerance
        ranked.append(
            (
                float(np.mean(camera_distance > tolerance)),
                int(np.count_nonzero(support)),
                normal,
                offset,
                support,
            )
        )
    positive_fraction, _, normal, offset, support = max(
        ranked, key=lambda item: (item[0], item[1])
    )
    return normal, offset, support, positive_fraction


def _court_transform(
    points: NDArray[np.float64],
    normal: NDArray[np.float64],
    offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    origin = np.median(points, axis=0)
    origin += (offset - float(origin @ normal)) * normal
    planar = points - origin
    planar -= np.outer(planar @ normal, normal)
    covariance = planar.T @ planar / len(planar)
    values, vectors = np.linalg.eigh(covariance)
    axes = [vectors[:, index] for index in np.argsort(values)[::-1][:2]]
    extents = [_percentile_extent(planar @ axis) for axis in axes]
    long_axis = axes[int(np.argmax(extents))]
    short_axis = axes[int(np.argmin(extents))]
    if short_axis[np.argmax(np.abs(short_axis))] < 0:
        short_axis = -short_axis
    longitudinal = np.cross(short_axis, normal)
    longitudinal /= np.linalg.norm(longitudinal)
    if float(longitudinal @ long_axis) < 0:
        longitudinal = -longitudinal
        short_axis = -short_axis
    rotation = np.column_stack([short_axis, normal, longitudinal])
    short_extent = _percentile_extent(planar @ short_axis)
    long_extent = _percentile_extent(planar @ longitudinal)
    scene_units_per_metre = min(
        short_extent / COURT_WIDTH_M,
        long_extent / COURT_LENGTH_M,
    )
    if not np.isfinite(scene_units_per_metre) or scene_units_per_metre <= 1.0e-8:
        raise ValueError("Could not infer a non-degenerate court scale")
    scene_from_court = np.eye(4, dtype=np.float64)
    scene_from_court[:3, :3] = rotation * scene_units_per_metre
    scene_from_court[:3, 3] = origin
    court_from_scene = np.asarray(np.linalg.inv(scene_from_court), dtype=np.float64)
    return (
        scene_from_court,
        court_from_scene,
        {
            "scene_units_per_metre": scene_units_per_metre,
            "support_short_extent_scene": short_extent,
            "support_long_extent_scene": long_extent,
        },
    )


def align_standard_scene(
    scene: StandardScene,
    output: Path,
    config: AlignmentConfig,
    seed: int,
) -> dict[str, Any]:
    """Fit a ground-aligned physical court frame and enforce semantic gates."""
    points = scene.points[:, :3].astype(np.float64)
    if len(points) < config.minimum_ground_points:
        raise ValueError("Point cloud is too small for ground alignment")
    random = np.random.default_rng(seed)
    if len(points) > 200_000:
        points = points[random.choice(len(points), 200_000, replace=False)]
    cameras = scene.camera_centers()
    if len(cameras) < 4:
        raise ValueError("At least four cameras are required for alignment")
    normal, offset, support, positive_camera_fraction = _ground_plane(points, cameras)
    support_points = points[support]
    support_fraction = float(len(support_points) / len(points))
    frame = plane_frame(support_points, normal, offset)
    distances = np.abs(support_points @ normal - offset)
    order = random.permutation(len(distances))
    holdout_count = max(1, round(len(order) * config.holdout_fraction))
    holdout = distances[order[:holdout_count]]
    fit = distances[order[holdout_count:]]
    fit_cameras, holdout_cameras = split_cameras(
        scene.cameras, config.evidence.maximum_views, config.holdout_fraction
    )
    fit_camera_ids = [str(camera["camera_id"]) for camera in fit_cameras]
    holdout_camera_ids = [str(camera["camera_id"]) for camera in holdout_cameras]
    if config.evidence.mode == "image_achromatic":
        fit_observations = observations(scene, fit_cameras, frame, config.evidence)
        holdout_observations = observations(
            scene, holdout_cameras, frame, config.evidence
        )
    else:
        fit_observations, holdout_observations = sparse_control_observations(
            support_points, frame, seed
        )
    if not fit_observations or not holdout_observations:
        raise ValueError("Court line extraction left fit or holdout evidence empty")
    line_arrays, grid_spacing = rasterize(
        fit_observations, frame, config.evidence.raster_size
    )
    parameters, template_fit = fit_court(
        line_arrays["mean_probability"],
        frame,
        grid_spacing,
        config.evidence,
        seed,
    )
    scene_from_court, court_from_scene = transforms(frame, parameters)
    fit_line_metrics = evaluate(fit_observations, court_from_scene, config.evidence)
    holdout_line_metrics = evaluate(
        holdout_observations, court_from_scene, config.evidence
    )
    gates = {
        "ground_support_count": {
            "value": len(support_points),
            "operator": ">=",
            "threshold": config.minimum_ground_points,
            "passed": len(support_points) >= config.minimum_ground_points,
        },
        "ground_support_fraction": {
            "value": support_fraction,
            "operator": ">=",
            "threshold": config.minimum_ground_support_fraction,
            "passed": support_fraction >= config.minimum_ground_support_fraction,
        },
        "positive_camera_fraction": {
            "value": positive_camera_fraction,
            "operator": ">=",
            "threshold": config.minimum_positive_camera_fraction,
            "passed": positive_camera_fraction
            >= config.minimum_positive_camera_fraction,
        },
        "fit_template_score": {
            "value": template_fit["normalized_template_score"],
            "operator": ">=",
            "threshold": config.evidence.minimum_fit_template_score,
            "passed": config.evidence.mode == "sparse_control"
            or template_fit["normalized_template_score"]
            >= config.evidence.minimum_fit_template_score,
        },
        "holdout_view_fraction": {
            "value": holdout_line_metrics["accepted_view_fraction"],
            "operator": ">=",
            "threshold": config.evidence.minimum_holdout_view_fraction,
            "passed": config.evidence.mode == "sparse_control"
            or holdout_line_metrics["accepted_view_fraction"]
            >= config.evidence.minimum_holdout_view_fraction,
        },
        "holdout_evaluable_view_count": {
            "value": holdout_line_metrics["evaluable_view_count"],
            "operator": ">=",
            "threshold": 1,
            "passed": config.evidence.mode == "sparse_control"
            or holdout_line_metrics["evaluable_view_count"] >= 1,
        },
        "holdout_line_inlier_fraction": {
            "value": holdout_line_metrics["weighted_inlier_fraction"],
            "operator": ">=",
            "threshold": config.evidence.minimum_holdout_inlier_fraction,
            "passed": config.evidence.mode == "sparse_control"
            or holdout_line_metrics["weighted_inlier_fraction"]
            >= config.evidence.minimum_holdout_inlier_fraction,
        },
    }
    accepted = all(gate["passed"] for gate in gates.values())
    payload: dict[str, Any] = {
        "schema": "tennis_scene_alignment_v1",
        "scene_id": scene.payload["scene_id"],
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "method": "held_out_ground_projected_court_line_template",
        "coordinate_convention": (
            "court x=lateral, y=up, z=longitudinal; metres; "
            "scene_from_court maps court metres into canonical NHT scene space"
        ),
        "scene_from_court": scene_from_court.tolist(),
        "court_from_scene": court_from_scene.tolist(),
        "ground_plane_scene": {
            "normal": normal.tolist(),
            "offset": offset,
        },
        "court_geometry_m": {
            "doubles_width": COURT_WIDTH_M,
            "length": COURT_LENGTH_M,
        },
        "fit_camera_ids": fit_camera_ids,
        "holdout_camera_ids": holdout_camera_ids,
        "fit_metrics": {
            "ground_residual_rms_scene": float(np.sqrt(np.mean(fit**2))),
            "ground_residual_p95_scene": float(np.percentile(fit, 95)),
            "court_lines": fit_line_metrics,
            "template": template_fit,
        },
        "holdout_metrics": {
            "ground_residual_rms_scene": float(np.sqrt(np.mean(holdout**2))),
            "ground_residual_p95_scene": float(np.percentile(holdout, 95)),
            "court_lines": holdout_line_metrics,
        },
        "support": {
            "point_count": len(support_points),
            "fraction": support_fraction,
            "positive_camera_fraction": positive_camera_fraction,
            "scene_units_per_metre": float(parameters[3]),
            "plane_bounds_uv": list(frame.bounds),
        },
        "gates": gates,
        "rejection_reasons": [
            name for name, gate in gates.items() if not gate["passed"]
        ],
        "settings": {
            "minimum_ground_points": config.minimum_ground_points,
            "minimum_ground_support_fraction": config.minimum_ground_support_fraction,
            "minimum_positive_camera_fraction": config.minimum_positive_camera_fraction,
            "holdout_fraction": config.holdout_fraction,
            "evidence": asdict(config.evidence),
        },
        "seed": seed,
    }
    geometry = {
        "schema": "tennis_court_geometry_v1",
        "scene_id": scene.payload["scene_id"],
        "status": payload["status"],
        "selected_court": {
            "doubles_width_m": COURT_WIDTH_M,
            "length_m": COURT_LENGTH_M,
            "scene_from_court": scene_from_court.tolist(),
            "court_from_scene": court_from_scene.tolist(),
            **template_fit,
        },
    }
    diagnostics = {
        "schema": "tennis_alignment_diagnostics_v1",
        "scene_id": scene.payload["scene_id"],
        "evidence_mode": config.evidence.mode,
        "fit_camera_ids": fit_camera_ids,
        "holdout_camera_ids": holdout_camera_ids,
        "fit_observation_camera_ids": [item.camera_id for item in fit_observations],
        "holdout_observation_camera_ids": [
            item.camera_id for item in holdout_observations
        ],
        "fit_metrics": fit_line_metrics,
        "holdout_metrics": holdout_line_metrics,
        "gates": gates,
    }
    publish_evidence(output.parent, line_arrays, grid_spacing, geometry, diagnostics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if not accepted:
        failures = [name for name, gate in gates.items() if not gate["passed"]]
        raise RuntimeError(f"Court alignment failed semantic gates: {failures}")
    return payload
