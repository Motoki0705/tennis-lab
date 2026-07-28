"""Estimate a deterministic ground plane from provider geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True)
class GroundPlaneFitSettings:
    """Frozen robust-fit settings in normalized provider scene units."""

    seed: int = 20260725
    footprint_quantile: float = 0.01
    footprint_margin: float = 0.5
    min_camera_height: float = 0.08
    max_camera_height: float = 0.30
    histogram_bin_width: float = 0.005
    candidate_half_width: float = 0.035
    ransac_threshold: float = 0.006
    refine_threshold: float = 0.008
    ransac_iterations: int = 3000
    ransac_sample_limit: int = 25000
    refine_iterations: int = 5
    min_candidate_points: int = 1000
    min_support_points: int = 10000
    min_normal_up_cosine: float = 0.98
    min_positive_camera_fraction: float = 1.0
    support_bounds_quantile: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("footprint_quantile", self.footprint_quantile),
            ("min_normal_up_cosine", self.min_normal_up_cosine),
            ("support_bounds_quantile", self.support_bounds_quantile),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {value}.")
        if not 0.0 <= self.min_positive_camera_fraction <= 1.0:
            raise ValueError(
                "min_positive_camera_fraction must lie in [0, 1], "
                f"got {self.min_positive_camera_fraction}."
            )
        for name, value in (
            ("footprint_margin", self.footprint_margin),
            ("min_camera_height", self.min_camera_height),
            ("max_camera_height", self.max_camera_height),
            ("histogram_bin_width", self.histogram_bin_width),
            ("candidate_half_width", self.candidate_half_width),
            ("ransac_threshold", self.ransac_threshold),
            ("refine_threshold", self.refine_threshold),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.min_camera_height >= self.max_camera_height:
            raise ValueError(
                "min_camera_height must be smaller than max_camera_height."
            )
        for name, value in (
            ("ransac_iterations", self.ransac_iterations),
            ("ransac_sample_limit", self.ransac_sample_limit),
            ("refine_iterations", self.refine_iterations),
            ("min_candidate_points", self.min_candidate_points),
            ("min_support_points", self.min_support_points),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class GroundPlaneEstimate:
    """Accepted oriented scene plane ``normal dot point + offset = 0``."""

    normal: tuple[float, float, float]
    offset: float
    origin: tuple[float, float, float]
    basis_u: tuple[float, float, float]
    basis_v: tuple[float, float, float]
    support_uv_bounds: tuple[float, float, float, float]
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        normal = _unit_vector(self.normal, name="normal")
        basis_u = _unit_vector(self.basis_u, name="basis_u")
        basis_v = _unit_vector(self.basis_v, name="basis_v")
        origin = _finite_vector(self.origin, name="origin")
        if abs(float(normal @ basis_u)) > 1.0e-6:
            raise ValueError("basis_u must lie in the ground plane.")
        if abs(float(normal @ basis_v)) > 1.0e-6:
            raise ValueError("basis_v must lie in the ground plane.")
        if not np.allclose(np.cross(basis_u, basis_v), normal, atol=1.0e-6):
            raise ValueError("basis_u, basis_v, normal must be right-handed.")
        offset = float(self.offset)
        if not np.isfinite(offset):
            raise ValueError("offset must be finite.")
        if abs(float(normal @ origin) + offset) > 1.0e-6:
            raise ValueError("origin must lie on the ground plane.")
        bounds = tuple(float(value) for value in self.support_uv_bounds)
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise ValueError("support_uv_bounds must contain four finite values.")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("support_uv_bounds must have positive area.")
        object.__setattr__(self, "normal", tuple(float(value) for value in normal))
        object.__setattr__(self, "basis_u", tuple(float(value) for value in basis_u))
        object.__setattr__(self, "basis_v", tuple(float(value) for value in basis_v))
        object.__setattr__(self, "origin", tuple(float(value) for value in origin))
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "support_uv_bounds", bounds)

    def signed_distance(
        self,
        points: NDArray[np.floating[Any]],
    ) -> NDArray[np.float64]:
        """Return oriented point-to-plane distances."""
        array = _points(points)
        normal = np.asarray(self.normal, dtype=np.float64)
        return array @ normal + self.offset

    def to_uv(
        self,
        points: NDArray[np.floating[Any]],
    ) -> NDArray[np.float64]:
        """Map scene points to the deterministic plane basis."""
        array = _points(points)
        origin = np.asarray(self.origin, dtype=np.float64)
        basis = np.stack((self.basis_u, self.basis_v), axis=1)
        return (array - origin) @ basis

    def from_uv(
        self,
        points_uv: NDArray[np.floating[Any]],
    ) -> NDArray[np.float64]:
        """Map plane coordinates back to scene coordinates."""
        array = np.asarray(points_uv, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != 2:
            raise ValueError(f"points_uv must have shape (..., 2), got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("points_uv must contain only finite values.")
        origin = np.asarray(self.origin, dtype=np.float64)
        basis = np.stack((self.basis_u, self.basis_v), axis=0)
        return array @ basis + origin

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "normal": list(self.normal),
            "offset": self.offset,
            "origin": list(self.origin),
            "basis_u": list(self.basis_u),
            "basis_v": list(self.basis_v),
            "support_uv_bounds": list(self.support_uv_bounds),
            "metrics": self.metrics,
        }


def estimate_ground_plane(
    points_scene: NDArray[np.floating[Any]],
    cameras: tuple[SceneCamera, ...],
    *,
    settings: GroundPlaneFitSettings,
) -> GroundPlaneEstimate:
    """Estimate the ground plane using fit cameras and nearby sparse points."""
    points = _points(points_scene)
    if len(points) < settings.min_candidate_points:
        raise ValueError("Point cloud is too small for ground-plane estimation.")
    if len(cameras) < 3:
        raise ValueError("At least three fit cameras are required.")

    poses = np.asarray(
        [camera.camera_to_scene for camera in cameras],
        dtype=np.float64,
    ).reshape(-1, 4, 4)
    camera_centers = poses[:, :3, 3]
    camera_ups = poses[:, :3, :3] @ np.asarray([0.0, -1.0, 0.0])
    up = np.mean(camera_ups, axis=0)
    up /= np.linalg.norm(up)
    up_cosines = camera_ups @ up
    if float(np.min(up_cosines)) < 0.90:
        raise ValueError("Fit-camera up vectors are not mutually consistent.")

    footprint_u = _project_axis_to_plane(
        np.asarray([1.0, 0.0, 0.0]),
        normal=up,
    )
    footprint_v = np.cross(up, footprint_u)
    footprint_basis = np.stack((footprint_u, footprint_v), axis=1)
    point_uv = points @ footprint_basis
    camera_uv = camera_centers @ footprint_basis
    low = (
        np.quantile(camera_uv, settings.footprint_quantile, axis=0)
        - settings.footprint_margin
    )
    high = (
        np.quantile(camera_uv, 1.0 - settings.footprint_quantile, axis=0)
        + settings.footprint_margin
    )
    footprint_mask = np.all((point_uv >= low) & (point_uv <= high), axis=1)

    point_heights = points @ up
    camera_height_coordinate = camera_centers @ up
    camera_median = float(np.median(camera_height_coordinate))
    eligible_mask = (
        footprint_mask
        & (point_heights >= camera_median - settings.max_camera_height)
        & (point_heights <= camera_median - settings.min_camera_height)
    )
    eligible_heights = point_heights[eligible_mask]
    if len(eligible_heights) < settings.min_candidate_points:
        raise ValueError("Too few ground-height candidate points.")
    edges = np.arange(
        camera_median - settings.max_camera_height,
        camera_median - settings.min_camera_height + settings.histogram_bin_width,
        settings.histogram_bin_width,
    )
    histogram, edges = np.histogram(eligible_heights, bins=edges)
    peak_index = int(np.argmax(histogram))
    peak_height = float((edges[peak_index] + edges[peak_index + 1]) / 2.0)
    candidate_mask = footprint_mask & (
        np.abs(point_heights - peak_height) <= settings.candidate_half_width
    )
    candidates = points[candidate_mask]
    if len(candidates) < settings.min_candidate_points:
        raise ValueError("Ground-height peak contains too few points.")

    normal, offset, ransac_inliers = _ransac_plane(
        candidates,
        up=up,
        settings=settings,
    )
    for _ in range(settings.refine_iterations):
        residuals = np.abs(candidates @ normal + offset)
        inliers = candidates[residuals <= settings.refine_threshold]
        if len(inliers) < settings.min_support_points:
            raise ValueError(
                "Ground-plane refinement has insufficient support: "
                f"{len(inliers)} < {settings.min_support_points}."
            )
        normal, offset = _fit_plane_svd(inliers, up=up)

    support_mask = footprint_mask & (
        np.abs(points @ normal + offset) <= settings.refine_threshold
    )
    support = points[support_mask]
    if len(support) < settings.min_support_points:
        raise ValueError("Accepted plane has insufficient point-cloud support.")
    normal_up_cosine = float(normal @ up)
    if normal_up_cosine < settings.min_normal_up_cosine:
        raise ValueError(
            "Ground-plane normal disagrees with fit-camera up direction: "
            f"{normal_up_cosine:.6f}."
        )
    camera_heights = camera_centers @ normal + offset
    positive_fraction = float(np.mean(camera_heights > 0.0))
    if positive_fraction < settings.min_positive_camera_fraction:
        raise ValueError(
            "Too many fit cameras lie on or below the ground plane: "
            f"positive fraction {positive_fraction:.6f}."
        )

    basis_u = _project_axis_to_plane(
        np.asarray([1.0, 0.0, 0.0]),
        normal=normal,
    )
    basis_v = np.cross(normal, basis_u)
    origin = -offset * normal
    support_uv = (support - origin) @ np.stack((basis_u, basis_v), axis=1)
    q = settings.support_bounds_quantile
    uv_low = np.quantile(support_uv, q, axis=0)
    uv_high = np.quantile(support_uv, 1.0 - q, axis=0)
    plane_residuals = support @ normal + offset
    metrics: dict[str, Any] = {
        "input_point_count": len(points),
        "footprint_point_count": int(footprint_mask.sum()),
        "eligible_height_point_count": len(eligible_heights),
        "height_histogram_peak": peak_height,
        "height_histogram_peak_count": int(histogram[peak_index]),
        "candidate_point_count": len(candidates),
        "ransac_inlier_count": ransac_inliers,
        "support_point_count": len(support),
        "support_residual_rms": float(np.sqrt(np.mean(np.square(plane_residuals)))),
        "normal_up_cosine": normal_up_cosine,
        "camera_up_cosine_min": float(np.min(up_cosines)),
        "camera_positive_height_fraction": positive_fraction,
        "camera_height_min": float(np.min(camera_heights)),
        "camera_height_q25": float(np.quantile(camera_heights, 0.25)),
        "camera_height_median": float(np.median(camera_heights)),
        "camera_height_q75": float(np.quantile(camera_heights, 0.75)),
        "camera_height_max": float(np.max(camera_heights)),
        "camera_height_std": float(np.std(camera_heights)),
        "reference_up": up.astype(float).tolist(),
        "settings": asdict(settings),
    }
    return GroundPlaneEstimate(
        normal=_triple(normal),
        offset=offset,
        origin=_triple(origin),
        basis_u=_triple(basis_u),
        basis_v=_triple(basis_v),
        support_uv_bounds=(
            float(uv_low[0]),
            float(uv_high[0]),
            float(uv_low[1]),
            float(uv_high[1]),
        ),
        metrics=metrics,
    )


def _ransac_plane(
    candidates: NDArray[np.float64],
    *,
    up: NDArray[np.float64],
    settings: GroundPlaneFitSettings,
) -> tuple[NDArray[np.float64], float, int]:
    rng = np.random.default_rng(settings.seed)
    sample_count = min(settings.ransac_sample_limit, len(candidates))
    sample = candidates[rng.choice(len(candidates), size=sample_count, replace=False)]
    best_score = -1
    best_median = np.inf
    best_normal: NDArray[np.float64] | None = None
    best_offset = 0.0
    for _ in range(settings.ransac_iterations):
        triplet = sample[rng.choice(len(sample), size=3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-10:
            continue
        normal /= norm
        if float(normal @ up) < 0.0:
            normal = -normal
        if float(normal @ up) < settings.min_normal_up_cosine:
            continue
        offset = -float(normal @ triplet[0])
        residuals = np.abs(sample @ normal + offset)
        mask = residuals <= settings.ransac_threshold
        score = int(mask.sum())
        median = float(np.median(residuals[mask])) if score else np.inf
        if score > best_score or (score == best_score and median < best_median):
            best_score = score
            best_median = median
            best_normal = normal.copy()
            best_offset = offset
    if best_normal is None:
        raise ValueError("Ground-plane RANSAC produced no valid hypothesis.")
    return best_normal, best_offset, best_score


def _fit_plane_svd(
    points: NDArray[np.float64],
    *,
    up: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    centroid = np.mean(points, axis=0)
    _, _, right = np.linalg.svd(points - centroid, full_matrices=False)
    normal = right[-1]
    if float(normal @ up) < 0.0:
        normal = -normal
    return normal, -float(normal @ centroid)


def _project_axis_to_plane(
    axis: NDArray[np.float64],
    *,
    normal: NDArray[np.float64],
) -> NDArray[np.float64]:
    projected = axis - normal * float(axis @ normal)
    if float(np.linalg.norm(projected)) < 0.1:
        fallback = np.asarray([0.0, 1.0, 0.0])
        projected = fallback - normal * float(fallback @ normal)
    projected /= np.linalg.norm(projected)
    return projected


def _points(
    value: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point array with shape (N, 3), got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Point array must contain only finite values.")
    return points


def _unit_vector(value: tuple[float, ...], *, name: str) -> NDArray[np.float64]:
    vector = _finite_vector(value, name=name)
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1.0e-6):
        raise ValueError(f"{name} must be unit length.")
    return vector


def _finite_vector(value: tuple[float, ...], *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values.")
    return vector


def _triple(value: NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))
