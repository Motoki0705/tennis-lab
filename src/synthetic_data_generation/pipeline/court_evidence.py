"""View-held-out court-line evidence and metric court-template fitting."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    maximum_filter,
    minimum_filter,
)
from scipy.optimize import differential_evolution

from .config import AlignmentEvidenceConfig
from .scene import StandardScene

COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77


@dataclass(frozen=True, slots=True)
class PlaneFrame:
    """One oriented ground plane and a right-handed in-plane basis."""

    normal: NDArray[np.float64]
    offset: float
    origin: NDArray[np.float64]
    basis_u: NDArray[np.float64]
    basis_v: NDArray[np.float64]
    bounds: tuple[float, float, float, float]

    def to_uv(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        basis = np.column_stack([self.basis_u, self.basis_v])
        return (points - self.origin) @ basis


@dataclass(frozen=True, slots=True)
class LineObservation:
    """Court-line pixels from one view after ground-plane projection."""

    camera_id: str
    points_scene: NDArray[np.float64]
    points_uv: NDArray[np.float64]
    scores: NDArray[np.float32]
    selected_pixel_count: int


def plane_frame(
    support_points: NDArray[np.float64],
    normal: NDArray[np.float64],
    offset: float,
) -> PlaneFrame:
    """Build a deterministic plane basis from accepted sparse support."""
    origin = np.median(support_points, axis=0)
    origin += (offset - float(origin @ normal)) * normal
    planar = support_points - origin
    planar -= np.outer(planar @ normal, normal)
    covariance = planar.T @ planar / len(planar)
    values, vectors = np.linalg.eigh(covariance)
    basis_u = vectors[:, int(np.argmax(values))]
    basis_u -= normal * float(basis_u @ normal)
    basis_u /= np.linalg.norm(basis_u)
    if basis_u[np.argmax(np.abs(basis_u))] < 0:
        basis_u = -basis_u
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    uv = planar @ np.column_stack([basis_u, basis_v])
    low = np.quantile(uv, 0.01, axis=0)
    high = np.quantile(uv, 0.99, axis=0)
    span = np.maximum(high - low, 1.0e-6)
    margin = 0.05 * span
    return PlaneFrame(
        normal=normal,
        offset=offset,
        origin=origin,
        basis_u=basis_u,
        basis_v=basis_v,
        bounds=(
            float(low[0] - margin[0]),
            float(high[0] + margin[0]),
            float(low[1] - margin[1]),
            float(high[1] + margin[1]),
        ),
    )


def split_cameras(
    cameras: Sequence[dict[str, Any]], maximum_views: int, holdout_fraction: float
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Choose time-spanning views and a deterministic interleaved holdout."""
    count = min(maximum_views, len(cameras))
    indices = np.linspace(0, len(cameras) - 1, count, dtype=np.int64)
    selected = tuple(cameras[int(index)] for index in indices)
    holdout_count = max(1, round(count * holdout_fraction))
    holdout_positions = set(
        int(index) for index in np.linspace(1, count - 1, holdout_count, dtype=np.int64)
    )
    holdout = tuple(
        camera for index, camera in enumerate(selected) if index in holdout_positions
    )
    fit = tuple(
        camera
        for index, camera in enumerate(selected)
        if index not in holdout_positions
    )
    if not fit or not holdout:
        raise ValueError("Court alignment requires non-empty fit and holdout views")
    return fit, holdout


def _line_pixels(
    image: Image.Image, config: AlignmentEvidenceConfig
) -> tuple[NDArray[np.float64], NDArray[np.float32], int]:
    original_width, original_height = image.size
    scale = min(1.0, config.maximum_image_size / max(image.size))
    width = max(2, round(original_width * scale))
    height = max(2, round(original_height * scale))
    resized = (
        np.asarray(
            image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
        )
        / 255.0
    )
    maximum = resized.max(axis=2)
    minimum = resized.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1.0e-6,
    )
    grey = resized.mean(axis=2)
    local_contrast = maximum_filter(grey, size=5) - minimum_filter(grey, size=5)
    selected = (
        (maximum >= config.minimum_line_brightness)
        & (saturation <= config.maximum_line_saturation)
        & (local_contrast >= config.minimum_local_contrast)
    )
    rows, columns = np.nonzero(selected)
    if len(rows) > config.maximum_pixels_per_view:
        order = np.argsort(local_contrast[rows, columns], kind="stable")
        keep = order[-config.maximum_pixels_per_view :]
        rows, columns = rows[keep], columns[keep]
    pixels = np.column_stack(
        [
            columns.astype(np.float64) * (original_width - 1) / (width - 1),
            rows.astype(np.float64) * (original_height - 1) / (height - 1),
        ]
    )
    scores = (
        maximum[rows, columns]
        * (1.0 - saturation[rows, columns])
        * np.clip(local_contrast[rows, columns] / 0.25, 0.0, 1.0)
    ).astype(np.float32)
    return pixels, scores, int(np.count_nonzero(selected))


def _project(
    camera: Mapping[str, Any],
    pixels: NDArray[np.float64],
    scores: NDArray[np.float32],
    frame: PlaneFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float32]]:
    intrinsic = np.asarray(camera["intrinsics"]["matrix"], dtype=np.float64)
    pose = np.asarray(camera["camera_to_scene"], dtype=np.float64)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    directions_camera = homogeneous @ np.linalg.inv(intrinsic).T
    directions_scene = directions_camera @ pose[:3, :3].T
    directions_scene /= np.linalg.norm(directions_scene, axis=1, keepdims=True)
    centre = pose[:3, 3]
    denominator = directions_scene @ frame.normal
    distance = np.divide(
        frame.offset - float(centre @ frame.normal),
        denominator,
        out=np.full(len(pixels), np.nan, dtype=np.float64),
        where=np.abs(denominator) >= 0.02,
    )
    points = centre + directions_scene * distance[:, None]
    uv = frame.to_uv(points)
    u_min, u_max, v_min, v_max = frame.bounds
    valid = (
        np.isfinite(distance)
        & (distance > 0.0)
        & (uv[:, 0] >= u_min)
        & (uv[:, 0] <= u_max)
        & (uv[:, 1] >= v_min)
        & (uv[:, 1] <= v_max)
    )
    return points[valid], uv[valid], scores[valid]


def observations(
    scene: StandardScene,
    cameras: Sequence[dict[str, Any]],
    frame: PlaneFrame,
    config: AlignmentEvidenceConfig,
) -> tuple[LineObservation, ...]:
    """Extract and ground-project explicit achromatic line evidence."""
    result = []
    for camera in cameras:
        with Image.open(scene.root / str(camera["image"])) as image:
            pixels, scores, selected_count = _line_pixels(image.convert("RGB"), config)
        points, uv, valid_scores = _project(camera, pixels, scores, frame)
        if len(points) < config.minimum_projected_pixels_per_view:
            continue
        result.append(
            LineObservation(
                camera_id=str(camera["camera_id"]),
                points_scene=points,
                points_uv=uv,
                scores=valid_scores,
                selected_pixel_count=selected_count,
            )
        )
    return tuple(result)


def sparse_control_observations(
    support_points: NDArray[np.float64], frame: PlaneFrame, seed: int
) -> tuple[tuple[LineObservation, ...], tuple[LineObservation, ...]]:
    """Provide an explicit model-free CPU control for orchestration tests."""
    random = np.random.default_rng(seed)
    order = random.permutation(len(support_points))
    middle = max(1, len(order) // 2)
    result = []
    for name, indices in (("fit", order[:middle]), ("holdout", order[middle:])):
        points = support_points[indices]
        result.append(
            (
                LineObservation(
                    camera_id=f"sparse-control-{name}",
                    points_scene=points,
                    points_uv=frame.to_uv(points),
                    scores=np.ones(len(points), dtype=np.float32),
                    selected_pixel_count=len(points),
                ),
            )
        )
    return result[0], result[1]


def rasterize(
    observations: Sequence[LineObservation],
    frame: PlaneFrame,
    raster_size: int,
) -> tuple[dict[str, NDArray[Any]], float]:
    """Accumulate at most one contribution per view and raster cell."""
    u_min, u_max, v_min, v_max = frame.bounds
    spacing = max(u_max - u_min, v_max - v_min) / (raster_size - 1)
    width = max(2, int(math.ceil((u_max - u_min) / spacing)) + 1)
    height = max(2, int(math.ceil((v_max - v_min) / spacing)) + 1)
    evidence_sum: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    weight_sum: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    view_count: NDArray[np.uint16] = np.zeros((height, width), dtype=np.uint16)
    for observation in observations:
        columns = np.rint((observation.points_uv[:, 0] - u_min) / spacing).astype(
            np.int64
        )
        rows = np.rint((observation.points_uv[:, 1] - v_min) / spacing).astype(np.int64)
        valid = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        flat = rows[valid] * width + columns[valid]
        per_view: NDArray[np.float32] = np.zeros(height * width, dtype=np.float32)
        np.maximum.at(per_view, flat, observation.scores[valid])
        occupied = per_view > 0.0
        evidence_sum.ravel()[occupied] += per_view[occupied]
        weight_sum.ravel()[occupied] += 1.0
        view_count.ravel()[occupied] += 1
    mean_probability = np.divide(
        evidence_sum,
        weight_sum,
        out=np.zeros_like(evidence_sum),
        where=weight_sum > 0.0,
    )
    return {
        "evidence_sum": evidence_sum,
        "weight_sum": weight_sum,
        "view_count": view_count,
        "mean_probability": mean_probability,
        "bounds": np.asarray(frame.bounds, dtype=np.float64),
    }, spacing


def _court_segments() -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    doubles = COURT_WIDTH_M / 2.0
    singles = 8.23 / 2.0
    baseline = COURT_LENGTH_M / 2.0
    service = 6.40
    return (
        ((-doubles, -baseline), (-doubles, baseline)),
        ((doubles, -baseline), (doubles, baseline)),
        ((-singles, -baseline), (-singles, baseline)),
        ((singles, -baseline), (singles, baseline)),
        ((-doubles, -baseline), (doubles, -baseline)),
        ((-doubles, baseline), (doubles, baseline)),
        ((-singles, -service), (singles, -service)),
        ((-singles, service), (singles, service)),
        ((0.0, -service), (0.0, service)),
    )


def _template(samples_per_metre: float = 5.0) -> NDArray[np.float64]:
    points = []
    for start, end in _court_segments():
        first = np.asarray(start, dtype=np.float64)
        second = np.asarray(end, dtype=np.float64)
        count = max(
            16, round(float(np.linalg.norm(second - first)) * samples_per_metre)
        )
        fraction = np.linspace(0.0, 1.0, count)[:, None]
        points.append(first * (1.0 - fraction) + second * fraction)
    return np.concatenate(points)


def _transform_template(
    template: NDArray[np.float64], parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    centre_u, centre_v, angle, scale = parameters
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    rotation_transpose = np.asarray(((cosine, sine), (-sine, cosine)))
    return template @ rotation_transpose * scale + np.asarray([centre_u, centre_v])


def fit_court(
    mean_probability: NDArray[np.float32],
    frame: PlaneFrame,
    spacing: float,
    config: AlignmentEvidenceConfig,
    seed: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Fit one ITF court template to fit-view line evidence."""
    if not bool(np.any(mean_probability > 0.0)):
        raise ValueError("Fit views produced no court-line raster evidence")
    score_image = gaussian_filter(np.log1p(mean_probability), sigma=1.5)
    template = _template()
    u_min, u_max, v_min, v_max = frame.bounds
    maximum_span = max(u_max - u_min, v_max - v_min)
    minimum_scale = max(maximum_span / 300.0, 1.0e-6)
    maximum_scale = maximum_span / 12.0

    def score(parameters: NDArray[np.float64]) -> float:
        uv = _transform_template(template, parameters)
        columns = (uv[:, 0] - u_min) / spacing
        rows = (uv[:, 1] - v_min) / spacing
        values = map_coordinates(
            score_image, [rows, columns], order=1, mode="constant", cval=0.0
        )
        return float(np.mean(values))

    result = differential_evolution(
        lambda parameters: -score(np.asarray(parameters, dtype=np.float64)),
        [
            (u_min, u_max),
            (v_min, v_max),
            (-math.pi / 2.0, math.pi / 2.0),
            (minimum_scale, maximum_scale),
        ],
        seed=seed,
        maxiter=config.optimizer_iterations,
        popsize=config.optimizer_population_size,
        tol=1.0e-7,
        polish=True,
        workers=1,
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    peak = max(float(np.max(score_image)), 1.0e-12)
    normalized_score = score(parameters) / peak
    return parameters, {
        "normalized_template_score": normalized_score,
        "raw_template_score": score(parameters),
        "optimizer_evaluations": int(result.nfev),
        "optimizer_success": bool(result.success),
        "parameters": {
            "center_uv": parameters[:2].tolist(),
            "orientation_radians": float(parameters[2]),
            "scene_units_per_metre": float(parameters[3]),
        },
    }


def transforms(
    frame: PlaneFrame, parameters: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build proper-handed x-lateral/y-up/z-longitudinal court transforms."""
    centre_u, centre_v, angle, scale = parameters
    lateral = (
        math.cos(float(angle)) * frame.basis_u + math.sin(float(angle)) * frame.basis_v
    )
    longitudinal = np.cross(lateral, frame.normal)
    longitudinal /= np.linalg.norm(longitudinal)
    translation = frame.origin + centre_u * frame.basis_u + centre_v * frame.basis_v
    scene_from_court = np.eye(4, dtype=np.float64)
    scene_from_court[:3, :3] = (
        np.column_stack([lateral, frame.normal, longitudinal]) * scale
    )
    scene_from_court[:3, 3] = translation
    if float(np.linalg.det(scene_from_court[:3, :3])) <= 0.0:
        raise ValueError("Court transform is not proper-handed")
    return scene_from_court, np.asarray(
        np.linalg.inv(scene_from_court), dtype=np.float64
    )


def _line_distances(points: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.full(len(points), np.inf, dtype=np.float64)
    for start, end in _court_segments():
        first = np.asarray(start, dtype=np.float64)
        delta = np.asarray(end, dtype=np.float64) - first
        fraction = np.clip(((points - first) @ delta) / float(delta @ delta), 0.0, 1.0)
        closest = first + fraction[:, None] * delta
        result = np.minimum(result, np.linalg.norm(points - closest, axis=1))
    return result


def evaluate(
    observations: Sequence[LineObservation],
    court_from_scene: NDArray[np.float64],
    config: AlignmentEvidenceConfig,
) -> dict[str, Any]:
    """Evaluate a fixed court transform without refitting held-out views."""
    by_view: dict[str, Any] = {}
    all_distances = []
    all_scores = []
    accepted = 0
    evaluable = 0
    for observation in observations:
        homogeneous = np.column_stack(
            [observation.points_scene, np.ones(len(observation.points_scene))]
        )
        court = (homogeneous @ court_from_scene.T)[:, :3]
        roi = (np.abs(court[:, 0]) <= COURT_WIDTH_M / 2.0 + 1.0) & (
            np.abs(court[:, 2]) <= COURT_LENGTH_M / 2.0 + 1.0
        )
        distance = _line_distances(court[roi][:, [0, 2]])
        scores = observation.scores[roi].astype(np.float64)
        if len(distance) and float(scores.sum()) > 0.0:
            inlier_fraction = float(
                scores[distance <= config.line_inlier_distance_m].sum() / scores.sum()
            )
            q95 = float(np.quantile(distance, 0.95))
        else:
            inlier_fraction, q95 = 0.0, None
        view_evaluable = len(distance) >= config.minimum_projected_pixels_per_view
        view_accepted = (
            view_evaluable and inlier_fraction >= config.minimum_holdout_inlier_fraction
        )
        evaluable += int(view_evaluable)
        accepted += int(view_accepted)
        by_view[observation.camera_id] = {
            "projected_line_points": len(observation.points_scene),
            "court_roi_points": len(distance),
            "weighted_inlier_fraction": inlier_fraction,
            "distance_q95_m": q95,
            "evaluable": view_evaluable,
            "accepted": view_accepted,
        }
        if view_evaluable:
            all_distances.append(distance)
            all_scores.append(scores)
    distances = np.concatenate(all_distances) if all_distances else np.empty(0)
    scores = np.concatenate(all_scores) if all_scores else np.empty(0)
    weighted_inlier = (
        float(scores[distances <= config.line_inlier_distance_m].sum() / scores.sum())
        if len(distances) and float(scores.sum()) > 0.0
        else 0.0
    )
    return {
        "view_count": len(observations),
        "evaluable_view_count": evaluable,
        "accepted_view_count": accepted,
        "accepted_view_fraction": accepted / evaluable if evaluable else 0.0,
        "weighted_inlier_fraction": weighted_inlier,
        "distance_q95_m": float(np.quantile(distances, 0.95))
        if len(distances)
        else None,
        "by_view": by_view,
    }


def publish_evidence(
    alignment_root: Path,
    arrays: dict[str, NDArray[Any]],
    spacing: float,
    geometry: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Publish the fixed mutable outputs owned by the alignment stage."""
    diagnostics_root = alignment_root / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        alignment_root / "ground-line-map.npz",
        grid_spacing=np.float64(spacing),
        **arrays,
    )
    evidence = arrays["mean_probability"]
    positive = evidence[evidence > 0.0]
    scale = float(np.quantile(positive, 0.995)) if len(positive) else 1.0
    preview = np.rint(np.clip(evidence / max(scale, 1.0e-8), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    Image.fromarray(np.flipud(preview), mode="L").save(
        alignment_root / "ground-line-preview.png"
    )
    (alignment_root / "court-geometry.json").write_text(
        json.dumps(geometry, indent=2) + "\n"
    )
    (diagnostics_root / "fit-holdout.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
