"""Fit calibration and holdout metrics for a fixed court-to-scene transform."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.synthetic_data_generation.alignment.court_template_fit import (
    court_line_segments,
    sample_court_line_template,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

ALIGNMENT_CALIBRATION_SCHEMA = "court_alignment_calibration_v1"
ALIGNMENT_VALIDATION_SCHEMA = "court_alignment_holdout_validation_v1"
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CourtLineEvaluationSettings:
    """Frozen metric settings applied identically to fit and holdout points."""

    line_inlier_distance_m: float = 0.25
    court_roi_margin_m: float = 1.0
    template_sample_spacing_m: float = 0.05
    point_cloud_vertical_tolerance_m: float = 0.15
    point_cloud_grid_spacing_m: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.line_inlier_distance_m <= 0.0
            or self.court_roi_margin_m < 0.0
            or self.template_sample_spacing_m <= 0.0
            or self.point_cloud_vertical_tolerance_m <= 0.0
            or self.point_cloud_grid_spacing_m <= 0.0
        ):
            raise ValueError("Court-line evaluation settings are invalid.")


def evaluate_projected_court_lines(
    points_scene: NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]],
    *,
    court_from_scene: NDArray[np.floating[Any]],
    settings: CourtLineEvaluationSettings,
) -> dict[str, Any]:
    """Measure fixed-transform line residuals and metric template coverage."""
    points_court = transform_points(points_scene, court_from_scene)
    weight = np.asarray(weights, dtype=np.float64)
    if weight.shape != (len(points_court),):
        raise ValueError("weights must have shape (N,).")
    if not np.isfinite(weight).all() or bool(np.any(weight < 0.0)):
        raise ValueError("weights must be finite and non-negative.")
    roi = court_roi_mask(
        points_court[:, :2],
        margin_m=settings.court_roi_margin_m,
    )
    selected = points_court[roi, :2]
    selected_weights = weight[roi]
    if len(selected) == 0 or float(selected_weights.sum()) <= 0.0:
        return {
            "input_point_count": len(points_court),
            "court_roi_point_count": 0,
            "weight_sum": 0.0,
            "weighted_inlier_fraction": 0.0,
            "raw_inlier_fraction": 0.0,
            "distance_weighted_q50_m": None,
            "distance_weighted_q90_m": None,
            "distance_weighted_q95_m": None,
            "distance_weighted_rms_m": None,
            "inlier_weighted_rms_m": None,
            "template_sample_count": 0,
            "template_coverage_fraction": 0.0,
            "template_distance_q95_m": None,
        }
    distances = court_line_distances(selected)
    inlier = distances <= settings.line_inlier_distance_m
    weight_sum = float(selected_weights.sum())
    inlier_weights = selected_weights[inlier]
    inlier_distances = distances[inlier]
    template = sample_court_line_template(1.0 / settings.template_sample_spacing_m)
    nearest_template_distance = cKDTree(selected).query(template)[0]
    return {
        "input_point_count": len(points_court),
        "court_roi_point_count": len(selected),
        "weight_sum": weight_sum,
        "weighted_inlier_fraction": float(selected_weights[inlier].sum() / weight_sum),
        "raw_inlier_fraction": float(np.mean(inlier)),
        "distance_weighted_q50_m": weighted_quantile(
            distances,
            selected_weights,
            0.5,
        ),
        "distance_weighted_q90_m": weighted_quantile(
            distances,
            selected_weights,
            0.9,
        ),
        "distance_weighted_q95_m": weighted_quantile(
            distances,
            selected_weights,
            0.95,
        ),
        "distance_weighted_rms_m": _weighted_rms(
            distances,
            selected_weights,
        ),
        "inlier_weighted_rms_m": (
            _weighted_rms(inlier_distances, inlier_weights)
            if len(inlier_distances)
            else None
        ),
        "template_sample_count": len(template),
        "template_coverage_fraction": float(
            np.mean(nearest_template_distance <= settings.line_inlier_distance_m)
        ),
        "template_distance_q95_m": float(np.quantile(nearest_template_distance, 0.95)),
    }


def point_cloud_court_support(
    points_scene: NDArray[np.floating[Any]],
    *,
    court_from_scene: NDArray[np.floating[Any]],
    settings: CourtLineEvaluationSettings,
) -> dict[str, Any]:
    """Measure independent point-cloud support for the selected court footprint."""
    points_court = transform_points(points_scene, court_from_scene)
    footprint = (np.abs(points_court[:, 0]) <= HALF_DOUBLES_WIDTH) & (
        np.abs(points_court[:, 1]) <= HALF_LENGTH
    )
    footprint_points = points_court[footprint]
    support = footprint_points[
        np.abs(footprint_points[:, 2]) <= settings.point_cloud_vertical_tolerance_m
    ]
    nx = int(np.ceil(2.0 * HALF_DOUBLES_WIDTH / settings.point_cloud_grid_spacing_m))
    ny = int(np.ceil(2.0 * HALF_LENGTH / settings.point_cloud_grid_spacing_m))
    if len(support):
        columns = np.floor(
            (support[:, 0] + HALF_DOUBLES_WIDTH) / settings.point_cloud_grid_spacing_m
        ).astype(np.int64)
        rows = np.floor(
            (support[:, 1] + HALF_LENGTH) / settings.point_cloud_grid_spacing_m
        ).astype(np.int64)
        valid = (columns >= 0) & (columns < nx) & (rows >= 0) & (rows < ny)
        counts = np.bincount(
            rows[valid] * nx + columns[valid],
            minlength=nx * ny,
        )
        absolute_height = np.abs(support[:, 2])
        residual_rms = float(np.sqrt(np.mean(np.square(support[:, 2]))))
        residual_q95 = float(np.quantile(absolute_height, 0.95))
    else:
        counts = np.zeros(nx * ny, dtype=np.int64)
        residual_rms = None
        residual_q95 = None
    return {
        "input_point_count": len(points_court),
        "footprint_point_count": len(footprint_points),
        "support_point_count": len(support),
        "vertical_tolerance_m": settings.point_cloud_vertical_tolerance_m,
        "residual_rms_m": residual_rms,
        "absolute_residual_q95_m": residual_q95,
        "grid_cell_count": nx * ny,
        "occupied_grid_fraction": float(np.mean(counts > 0)),
        "occupied_three_point_grid_fraction": float(np.mean(counts >= 3)),
    }


def camera_heights_in_court(
    camera_centres_scene: NDArray[np.floating[Any]],
    *,
    court_from_scene: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return fixed-camera heights in metric court coordinates."""
    return transform_points(camera_centres_scene, court_from_scene)[:, 2]


def holdout_gate_results(
    metrics: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, bool]:
    """Apply every frozen aggregate, per-group, and camera-height gate."""
    aggregate = metrics["aggregate"]
    by_group = metrics["by_group"]
    heights = metrics["camera_heights_m"]
    return {
        "accepted_view_fraction": metrics["accepted_view_fraction"]
        >= gates["minimum_accepted_view_fraction"],
        "weighted_inlier_fraction": aggregate["weighted_inlier_fraction"]
        >= gates["minimum_weighted_inlier_fraction"],
        "distance_weighted_q95": aggregate["distance_weighted_q95_m"]
        <= gates["maximum_distance_weighted_q95_m"],
        "template_coverage": aggregate["template_coverage_fraction"]
        >= gates["minimum_template_coverage_fraction"],
        "every_group_weighted_inlier": all(
            item["weighted_inlier_fraction"]
            >= gates["minimum_group_weighted_inlier_fraction"]
            for item in by_group.values()
        ),
        "every_group_template_coverage": all(
            item["template_coverage_fraction"]
            >= gates["minimum_group_template_coverage_fraction"]
            for item in by_group.values()
        ),
        "every_group_accepted_views": all(
            count >= gates["minimum_accepted_views_per_group"]
            for count in metrics["accepted_view_count_by_group"].values()
        ),
        "camera_height_minimum": heights["minimum"] >= gates["minimum_camera_height_m"],
        "camera_height_maximum": heights["maximum"] <= gates["maximum_camera_height_m"],
        "camera_height_positive_fraction": heights["positive_fraction"]
        >= gates["minimum_positive_camera_height_fraction"],
    }


def transform_stability(
    reference_scene_from_court: NDArray[np.floating[Any]],
    candidate_scene_from_court: NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Compare two proper Sim(3) court transforms modulo 180-degree court symmetry."""
    reference = _similarity_matrix(reference_scene_from_court)
    candidate = _similarity_matrix(candidate_scene_from_court)
    reference_scale = float(np.linalg.norm(reference[:3, 0]))
    candidate_scale = float(np.linalg.norm(candidate[:3, 0]))
    reference_rotation = reference[:3, :3] / reference_scale
    candidate_rotation = candidate[:3, :3] / candidate_scale
    relative_rotation = reference_rotation.T @ candidate_rotation
    angle = float(
        np.degrees(
            np.arccos(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        )
    )
    angle = min(angle, abs(180.0 - angle))
    reference_inverse = np.linalg.inv(reference)
    candidate_origin_reference = reference_inverse @ np.asarray(
        [
            candidate[0, 3],
            candidate[1, 3],
            candidate[2, 3],
            1.0,
        ]
    )
    return {
        "centre_shift_m": float(np.linalg.norm(candidate_origin_reference[:3])),
        "orientation_difference_deg_mod_180": angle,
        "relative_scale_difference": abs(candidate_scale - reference_scale)
        / reference_scale,
    }


def court_line_distances(
    points_court_xy: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return nearest finite-segment distance to the painted ITF template."""
    points = np.asarray(points_court_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_court_xy must have shape (N, 2).")
    if not np.isfinite(points).all():
        raise ValueError("points_court_xy must contain only finite values.")
    result: NDArray[np.float64] = np.full(
        len(points),
        np.inf,
        dtype=np.float64,
    )
    for start, end in court_line_segments():
        start_array = np.asarray(start, dtype=np.float64)
        delta = np.asarray(end, dtype=np.float64) - start_array
        fraction = np.clip(
            ((points - start_array) @ delta) / float(delta @ delta),
            0.0,
            1.0,
        )
        closest = start_array + fraction[:, None] * delta
        result = np.minimum(result, np.linalg.norm(points - closest, axis=1))
    return result


def court_roi_mask(
    points_court_xy: NDArray[np.floating[Any]],
    *,
    margin_m: float,
) -> NDArray[np.bool_]:
    """Select a physical court instance without admitting an adjacent court."""
    points = np.asarray(points_court_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_court_xy must have shape (N, 2).")
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative.")
    return (np.abs(points[:, 0]) <= HALF_DOUBLES_WIDTH + margin_m) & (
        np.abs(points[:, 1]) <= HALF_LENGTH + margin_m
    )


def transform_points(
    points: NDArray[np.floating[Any]],
    transform: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Apply a finite homogeneous transform to 3D points."""
    array = np.asarray(points, dtype=np.float64)
    matrix = _similarity_matrix(transform)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(array).all():
        raise ValueError("points must contain only finite values.")
    homogeneous = np.column_stack((array, np.ones(len(array))))
    return (homogeneous @ matrix.T)[:, :3]


def weighted_quantile(
    values: NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]],
    quantile: float,
) -> float:
    """Return a deterministic positive-weight empirical quantile."""
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if value.shape != weight.shape or value.ndim != 1:
        raise ValueError("values and weights must be same-shape 1D arrays.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not bool(np.any(valid)):
        raise ValueError("At least one finite positive weight is required.")
    order = np.argsort(value[valid], kind="stable")
    sorted_values = value[valid][order]
    cumulative = np.cumsum(weight[valid][order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def publish_alignment_artifact(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    """Atomically publish a strict fingerprinted calibration or validation JSON."""
    _validate_payload(payload)
    manifest = dict(payload)
    fingerprint = _canonical_fingerprint(manifest)
    manifest["artifact_fingerprint"] = fingerprint
    destination = output_dir / f"{payload['artifact_id']}-{fingerprint[:16]}.json"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite alignment artifact: {destination}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{payload['artifact_id']}-",
        suffix=".json",
        dir=output_dir,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.rename(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_alignment_artifact(path: Path) -> dict[str, Any]:
    """Load and fingerprint-verify one C05 alignment artifact."""
    with path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Alignment artifact must be a JSON object.")
    payload = dict(raw)
    _validate_payload(payload)
    declared = payload.get("artifact_fingerprint")
    expected = _canonical_fingerprint(payload)
    if declared != expected:
        raise ValueError(
            "Alignment artifact fingerprint mismatch: "
            f"declared {declared}, computed {expected}."
        )
    return payload


def _weighted_rms(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    return float(np.sqrt(np.average(np.square(values), weights=weights)))


def _similarity_matrix(
    value: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Similarity transform must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError("Similarity transform must be homogeneous.")
    scales = np.linalg.norm(matrix[:3, :3], axis=0)
    if bool(np.any(scales <= 0.0)) or not np.allclose(
        scales, scales[0], atol=1.0e-8, rtol=1.0e-6
    ):
        raise ValueError("Similarity transform must have one positive scale.")
    rotation = matrix[:3, :3] / scales[0]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("Similarity transform rotation must be orthonormal.")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("Similarity transform must reject reflection.")
    return matrix


def _validate_payload(payload: dict[str, Any]) -> None:
    schema = payload.get("schema")
    if schema not in {ALIGNMENT_CALIBRATION_SCHEMA, ALIGNMENT_VALIDATION_SCHEMA}:
        raise ValueError(f"Unsupported alignment artifact schema: {schema!r}.")
    artifact_id = payload.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or _ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None
    ):
        raise ValueError("Alignment artifact_id must be path-safe.")
    common = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "provider",
        "geometry",
        "split",
        "evaluation_settings",
        "gates",
        "metrics",
        "gate_results",
        "status",
        "provenance",
    }
    if schema == ALIGNMENT_CALIBRATION_SCHEMA:
        required = common | {"detector", "stability", "point_cloud_support"}
        if payload.get("split", {}).get("holdout_inference_status") != "not_run":
            raise ValueError("Calibration must not infer holdout images.")
        if payload.get("status") not in {
            "fit_calibration_passed",
            "fit_calibration_failed",
        }:
            raise ValueError("Invalid calibration status.")
    else:
        required = common | {"calibration", "detector", "records"}
        if payload.get("split", {}).get("holdout_inference_status") != "complete":
            raise ValueError("Validation must record completed holdout inference.")
        if payload.get("status") not in {
            "accepted",
            "rejected",
        }:
            raise ValueError("Invalid holdout validation status.")
    optional = {"artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Alignment artifact keys do not match its schema.")


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_fingerprint", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
