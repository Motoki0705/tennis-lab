"""Bounded public-camera support and public-point occupancy for Court V4."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitPathSamples,
    SupportModelSummary,
    TrajectorySafetyEvaluation,
    TrajectorySafetyReason,
    TrajectorySupportPolicy,
)
from src.synthetic_data_generation.scene_contract import SceneCamera

Cell = tuple[int, int, int]


class TrajectorySupportError(ValueError):
    """Fail-closed support construction error with one stable reason code."""

    def __init__(self, reason: TrajectorySafetyReason, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason.value}: {detail}")


@dataclass(frozen=True, slots=True)
class _SupportPrimitive:
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True, slots=True)
class TrajectorySupportModel:
    """Immutable bounded indices used by planning and pre-render revalidation."""

    policy: TrajectorySupportPolicy
    summary: SupportModelSummary
    primitives: tuple[_SupportPrimitive, ...]
    support_index: Mapping[Cell, tuple[int, ...]]
    inflated_occupancy: frozenset[Cell]
    occupancy_centers_m: NDArray[np.float64]
    occupancy_index: cKDTree
    captured_cameras: tuple[SceneCamera, ...]
    captured_camera_inventory_digest: str
    captured_camera_centers_m: NDArray[np.float64]
    trusted_temporal_links: tuple[tuple[int, int], ...]

    def evaluate_point(
        self, point_scene_m: NDArray[np.floating]
    ) -> tuple[float, float, bool, bool]:
        """Return support margin, obstacle clearance, support, and occupancy decisions."""
        point = np.asarray(point_scene_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Support query point must be one finite 3-vector.")
        support_margin, supported = self._evaluate_support(point)
        occupied_cell = (
            _cell(point, self.policy.occupancy_voxel_size_m) in self.inflated_occupancy
        )
        clearance = self._obstacle_clearance(point, occupied=occupied_cell)
        occupied = occupied_cell or clearance <= self.policy.boundary_epsilon_m
        return support_margin, clearance, supported, occupied

    def _evaluate_support(self, point: NDArray[np.float64]) -> tuple[float, bool]:
        support_cell = _cell(point, self.policy.support_radius_m)
        candidates = self.support_index.get(support_cell, ())
        support_margin = -self.policy.support_radius_m
        for index in candidates:
            primitive = self.primitives[index]
            distance = _point_segment_distance(
                point,
                np.asarray(primitive.start_m, dtype=np.float64),
                np.asarray(primitive.end_m, dtype=np.float64),
            )
            support_margin = max(support_margin, primitive.radius_m - distance)
        supported = support_margin > self.policy.boundary_epsilon_m
        return support_margin, supported

    def segment_is_safe(
        self,
        start_scene_m: NDArray[np.floating],
        end_scene_m: NDArray[np.floating],
    ) -> bool:
        """Return exact residual-occupancy and sampled support authority."""
        start = np.asarray(start_scene_m, dtype=np.float64)
        end = np.asarray(end_scene_m, dtype=np.float64)
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or not np.isfinite(start).all()
            or not np.isfinite(end).all()
        ):
            raise ValueError("Support segment endpoints must be finite 3-vectors.")
        if _segment_hits_occupancy(
            start,
            end,
            inflated=self.inflated_occupancy,
            policy=self.policy,
        ):
            return False
        distance = float(np.linalg.norm(end - start))
        count = max(1, math.ceil(distance / self.policy.sweep_step_m))
        return all(
            self._evaluate_support(start + (end - start) * (index / count))[1]
            for index in range(count + 1)
        )

    def _obstacle_clearance(
        self, point: NDArray[np.float64], *, occupied: bool
    ) -> float:
        if occupied:
            return -self.policy.boundary_epsilon_m
        voxel = self.policy.occupancy_voxel_size_m
        half_voxel = voxel / 2.0
        half_diagonal = math.sqrt(3.0) * half_voxel
        cap = self.policy.support_radius_m
        nearest_distance, nearest_index = self.occupancy_index.query(
            point,
            k=1,
            distance_upper_bound=cap + half_diagonal,
            workers=1,
        )
        nearest_index_int = int(nearest_index)
        if not math.isfinite(float(nearest_distance)) or nearest_index_int >= len(
            self.occupancy_centers_m
        ):
            return cap
        nearest_center = self.occupancy_centers_m[
            nearest_index_int : nearest_index_int + 1
        ]
        upper_bound = min(
            cap,
            _minimum_box_clearance(
                point,
                centers=nearest_center,
                half_extent=half_voxel,
            ),
        )
        search_radius = float(np.nextafter(upper_bound + half_diagonal, math.inf))
        candidate_indices = np.asarray(
            self.occupancy_index.query_ball_point(
                point,
                r=search_radius,
                eps=0.0,
                workers=1,
            ),
            dtype=np.int64,
        )
        if candidate_indices.size == 0:
            return cap
        return min(
            cap,
            _minimum_box_clearance(
                point,
                centers=self.occupancy_centers_m[candidate_indices],
                half_extent=half_voxel,
            ),
        )


def build_trajectory_support_model(
    *,
    cameras: Sequence[SceneCamera],
    points_scene_m: NDArray[np.floating],
    policy: TrajectorySupportPolicy,
) -> TrajectorySupportModel:
    """Build one bounded model from already metric-transformed public inputs."""
    if not isinstance(policy, TrajectorySupportPolicy):
        raise TypeError("policy must be a TrajectorySupportPolicy.")
    camera_tuple = canonical_public_camera_inventory(cameras)
    if len(camera_tuple) < policy.minimum_captured_cameras:
        raise TrajectorySupportError(
            TrajectorySafetyReason.INSUFFICIENT_CAPTURED_CAMERAS,
            f"required={policy.minimum_captured_cameras}, observed={len(camera_tuple)}",
        )
    centers = np.asarray(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in camera_tuple],
        dtype=np.float64,
    )
    points = np.asarray(points_scene_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise TrajectorySupportError(
            TrajectorySafetyReason.MISSING_SUPPORT_CAPABILITY,
            "public points must have shape (N,3) or (N,6)",
        )
    points = points[:, :3]
    if not np.isfinite(centers).all() or not np.isfinite(points).all():
        raise TrajectorySupportError(
            TrajectorySafetyReason.NONFINITE_SUPPORT_INPUT,
            "captured cameras and public points must be finite",
        )
    if len(points) < policy.minimum_public_points:
        raise TrajectorySupportError(
            TrajectorySafetyReason.INSUFFICIENT_PUBLIC_POINTS,
            f"required={policy.minimum_public_points}, observed={len(points)}",
        )
    qualified = _density_qualified_cells(points, policy=policy)
    raw_inflated = _inflate_occupancy(qualified, policy=policy)
    ordered = tuple(zip(camera_tuple, centers, strict=True))
    trusted_links: list[tuple[int, int]] = []
    skipped_gap = 0
    for index, ((previous_camera, previous), (current_camera, current)) in enumerate(
        zip(ordered, ordered[1:], strict=False)
    ):
        frame_gap = (
            current_camera.source_frame_index - previous_camera.source_frame_index
        )
        distance = float(np.linalg.norm(current - previous))
        if (
            frame_gap <= 0
            or frame_gap > policy.maximum_source_frame_gap
            or distance > policy.maximum_camera_link_distance_m
        ):
            skipped_gap += 1
            continue
        trusted_links.append((index, index + 1))
    after_balls = _carve_occupancy(
        raw_inflated,
        segments=((center, center) for _camera, center in ordered),
        radius_m=policy.camera_ball_clearance_m,
        policy=policy,
    )
    inflated = _carve_occupancy(
        after_balls,
        segments=((ordered[start][1], ordered[end][1]) for start, end in trusted_links),
        radius_m=policy.camera_capsule_clearance_m,
        policy=policy,
    )
    if not inflated:
        raise TrajectorySupportError(
            TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE,
            "camera carving removed every inflated occupancy cell",
        )
    endpoint_primitives = tuple(
        _SupportPrimitive(
            start_m=(float(center[0]), float(center[1]), float(center[2])),
            end_m=(float(center[0]), float(center[1]), float(center[2])),
            radius_m=policy.endpoint_radius_m,
        )
        for _camera, center in ordered
    )
    capsules: list[_SupportPrimitive] = []
    skipped_obstacle = 0
    for start_index, end_index in trusted_links:
        previous = ordered[start_index][1]
        current = ordered[end_index][1]
        if _segment_hits_occupancy(previous, current, inflated=inflated, policy=policy):
            raise TrajectorySupportError(
                TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE,
                "a trusted camera link intersects residual occupancy after carving",
            )
        capsules.append(
            _SupportPrimitive(
                start_m=(
                    float(previous[0]),
                    float(previous[1]),
                    float(previous[2]),
                ),
                end_m=(
                    float(current[0]),
                    float(current[1]),
                    float(current[2]),
                ),
                radius_m=policy.support_radius_m,
            )
        )
    if not capsules:
        raise TrajectorySupportError(
            TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE,
            "no adjacent captured-camera link survived gap and obstacle checks",
        )
    primitives = (*endpoint_primitives, *capsules)
    support_index = _build_support_index(primitives, policy=policy)
    digest = _support_input_digest(
        ordered=ordered,
        points=points,
        policy=policy,
    )
    captured_occupied = sum(
        _cell(center, policy.occupancy_voxel_size_m) in inflated
        for _camera, center in ordered
    )
    summary = SupportModelSummary(
        input_digest=digest,
        coordinate_space="metric_scene_metres",
        captured_camera_count=len(camera_tuple),
        public_point_count=len(points),
        density_qualified_voxel_count=len(qualified),
        raw_inflated_occupancy_cell_count=len(raw_inflated),
        inflated_occupancy_cell_count=len(inflated),
        camera_ball_carved_cell_count=len(raw_inflated) - len(after_balls),
        camera_capsule_carved_cell_count=len(after_balls) - len(inflated),
        captured_camera_occupied_count=captured_occupied,
        endpoint_ball_count=len(endpoint_primitives),
        capsule_count=len(capsules),
        skipped_gap_link_count=skipped_gap,
        skipped_obstacle_link_count=skipped_obstacle,
        capsule_index_cell_count=len(support_index),
    )
    occupancy_centers, occupancy_index = _build_occupancy_index(
        inflated,
        policy=policy,
    )
    return TrajectorySupportModel(
        policy=policy,
        summary=summary,
        primitives=primitives,
        support_index=support_index,
        inflated_occupancy=inflated,
        occupancy_centers_m=occupancy_centers,
        occupancy_index=occupancy_index,
        captured_cameras=camera_tuple,
        captured_camera_inventory_digest=public_camera_inventory_digest(
            camera_tuple
        ),
        captured_camera_centers_m=_readonly_centers(ordered),
        trusted_temporal_links=tuple(trusted_links),
    )


def canonical_public_camera_inventory(
    cameras: Sequence[SceneCamera],
) -> tuple[SceneCamera, ...]:
    """Return the one typed order used by support and anchor provenance."""
    values = tuple(cameras)
    if not values or any(not isinstance(camera, SceneCamera) for camera in values):
        raise TypeError("Public camera inventory must be non-empty and typed.")
    camera_ids = tuple(camera.camera_id for camera in values)
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("Public camera inventory IDs must be unique.")
    return tuple(
        sorted(
            values,
            key=lambda camera: (camera.source_frame_index, camera.camera_id),
        )
    )


def public_camera_inventory_digest(cameras: Sequence[SceneCamera]) -> str:
    """Hash every public camera in the canonical support-authority order."""
    ordered = canonical_public_camera_inventory(cameras)
    payload = json.dumps(
        [camera.to_dict() for camera in ordered],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_trajectory_safety(
    *,
    trajectory_id: str,
    trajectory_group_id: str,
    path: OrbitPathSamples,
    support_model: TrajectorySupportModel,
) -> TrajectorySafetyEvaluation:
    """Evaluate every point and every subdivided closed edge, including the seam."""
    points = np.asarray(path.points_scene_m, dtype=np.float64)
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError(
            "Safety evaluation requires a finite closed path with >=8 points."
        )
    reasons: set[TrajectorySafetyReason] = set()
    violating_points: list[int] = []
    violating_segments: list[int] = []
    support_margins: list[float] = []
    obstacle_clearances: list[float] = []
    for index, point in enumerate(points):
        support_margin, obstacle_clearance, supported, occupied = (
            support_model.evaluate_point(point)
        )
        support_margins.append(support_margin)
        obstacle_clearances.append(obstacle_clearance)
        if not supported:
            reasons.add(TrajectorySafetyReason.POINT_OUTSIDE_SUPPORT)
            violating_points.append(index)
        if occupied:
            reasons.add(TrajectorySafetyReason.POINT_HITS_INFLATED_OBSTACLE)
            if index not in violating_points:
                violating_points.append(index)
    swept_count = 0
    for segment_index, (start, end) in enumerate(
        zip(points, np.roll(points, -1, axis=0), strict=True)
    ):
        distance = float(np.linalg.norm(end - start))
        subdivisions = max(1, math.ceil(distance / support_model.policy.sweep_step_m))
        segment_occupied = _segment_hits_occupancy(
            start,
            end,
            inflated=support_model.inflated_occupancy,
            policy=support_model.policy,
        )
        segment_invalid = segment_occupied
        if segment_occupied:
            reasons.add(TrajectorySafetyReason.SWEPT_SEGMENT_HITS_INFLATED_OBSTACLE)
            obstacle_clearances.append(-support_model.policy.boundary_epsilon_m)
        for offset in range(1, subdivisions + 1):
            point = start + (end - start) * (offset / subdivisions)
            support_margin, supported = support_model._evaluate_support(point)
            support_margins.append(support_margin)
            swept_count += 1
            if not supported:
                reasons.add(TrajectorySafetyReason.SWEPT_SEGMENT_OUTSIDE_SUPPORT)
                segment_invalid = True
        if segment_invalid:
            violating_segments.append(segment_index)
    ordered_reasons = tuple(
        reason for reason in TrajectorySafetyReason if reason in reasons
    )
    return TrajectorySafetyEvaluation(
        trajectory_id=trajectory_id,
        trajectory_group_id=trajectory_group_id,
        support_input_digest=support_model.summary.input_digest,
        safe=not ordered_reasons,
        reasons=ordered_reasons,
        path_point_count=len(points),
        closed_segment_count=len(points),
        swept_sample_count=swept_count,
        violating_point_indices=tuple(sorted(violating_points)),
        violating_segment_indices=tuple(violating_segments),
        minimum_support_margin_m=min(support_margins),
        minimum_obstacle_clearance_m=min(obstacle_clearances),
    )


def _density_qualified_cells(
    points: NDArray[np.float64], *, policy: TrajectorySupportPolicy
) -> frozenset[Cell]:
    indices = np.floor(points / policy.occupancy_voxel_size_m).astype(np.int64)
    counts: Counter[Cell] = Counter(
        (int(row[0]), int(row[1]), int(row[2])) for row in indices
    )
    qualified = frozenset(
        cell
        for cell, count in counts.items()
        if count >= policy.minimum_points_per_voxel
    )
    if not qualified:
        raise TrajectorySupportError(
            TrajectorySafetyReason.INSUFFICIENT_PUBLIC_POINTS,
            "density qualification produced zero occupied voxels",
        )
    return qualified


def _inflate_occupancy(
    occupied: frozenset[Cell], *, policy: TrajectorySupportPolicy
) -> frozenset[Cell]:
    voxel = policy.occupancy_voxel_size_m
    radius = math.ceil(policy.obstacle_inflation_m / voxel) + 1
    offsets: list[Cell] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                separation = np.maximum(np.abs((dx, dy, dz)) - 1, 0) * voxel
                if float(np.linalg.norm(separation)) <= (
                    policy.obstacle_inflation_m + policy.boundary_epsilon_m
                ):
                    offsets.append((dx, dy, dz))
    result: set[Cell] = set()
    for cell in occupied:
        for offset in offsets:
            result.add((cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2]))
            if len(result) > policy.maximum_occupancy_cells:
                raise TrajectorySupportError(
                    TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE,
                    "inflated occupancy exceeds maximum_occupancy_cells",
                )
    return frozenset(result)


def _segment_hits_occupancy(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    *,
    inflated: frozenset[Cell],
    policy: TrajectorySupportPolicy,
) -> bool:
    voxel = policy.occupancy_voxel_size_m
    epsilon = policy.boundary_epsilon_m
    lower = np.floor((np.minimum(start, end) - epsilon) / voxel).astype(np.int64)
    upper = np.floor((np.maximum(start, end) + epsilon) / voxel).astype(np.int64)
    for x in range(int(lower[0]), int(upper[0]) + 1):
        for y in range(int(lower[1]), int(upper[1]) + 1):
            for z in range(int(lower[2]), int(upper[2]) + 1):
                cell = (x, y, z)
                if cell not in inflated:
                    continue
                box_lower = np.asarray(cell, dtype=np.float64) * voxel
                box_upper = box_lower + voxel
                if _segment_intersects_aabb(
                    start,
                    end,
                    lower=box_lower,
                    upper=box_upper,
                    epsilon=epsilon,
                ):
                    return True
    return False


def _carve_occupancy(
    occupied: frozenset[Cell],
    *,
    segments: Iterable[tuple[NDArray[np.float64], NDArray[np.float64]]],
    radius_m: float,
    policy: TrajectorySupportPolicy,
) -> frozenset[Cell]:
    """Subtract exact Euclidean balls/capsules from inflated occupancy cells."""
    residual = set(occupied)
    voxel = policy.occupancy_voxel_size_m
    epsilon = policy.boundary_epsilon_m
    for start_raw, end_raw in segments:
        start = np.asarray(start_raw, dtype=np.float64)
        end = np.asarray(end_raw, dtype=np.float64)
        lower = np.floor((np.minimum(start, end) - radius_m) / voxel).astype(np.int64)
        upper = np.floor((np.maximum(start, end) + radius_m) / voxel).astype(np.int64)
        for x in range(int(lower[0]), int(upper[0]) + 1):
            for y in range(int(lower[1]), int(upper[1]) + 1):
                for z in range(int(lower[2]), int(upper[2]) + 1):
                    cell = (x, y, z)
                    if cell not in residual:
                        continue
                    box_lower = np.asarray(cell, dtype=np.float64) * voxel
                    box_upper = box_lower + voxel
                    if (
                        _segment_aabb_distance_squared(
                            start,
                            end,
                            lower=box_lower,
                            upper=box_upper,
                        )
                        <= (radius_m + epsilon) ** 2
                    ):
                        residual.remove(cell)
    return frozenset(residual)


def _segment_intersects_aabb(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    epsilon: float,
) -> bool:
    direction = end - start
    minimum = 0.0
    maximum = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= 1.0e-15:
            if (
                start[axis] < lower[axis] - epsilon
                or start[axis] > upper[axis] + epsilon
            ):
                return False
            continue
        first = (lower[axis] - epsilon - start[axis]) / direction[axis]
        second = (upper[axis] + epsilon - start[axis]) / direction[axis]
        entry = min(float(first), float(second))
        exit_ = max(float(first), float(second))
        minimum = max(minimum, entry)
        maximum = min(maximum, exit_)
        if minimum > maximum:
            return False
    return True


def _segment_aabb_distance_squared(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    *,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Return the exact squared Euclidean distance from a segment to an AABB."""
    direction = end - start
    breakpoints = {0.0, 1.0}
    for axis in range(3):
        if abs(float(direction[axis])) <= 1.0e-15:
            continue
        for boundary in (lower[axis], upper[axis]):
            value = float((boundary - start[axis]) / direction[axis])
            if 0.0 < value < 1.0:
                breakpoints.add(value)
    ordered = sorted(breakpoints)

    def distance_squared(parameter: float) -> float:
        point = start + direction * parameter
        delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        return float(delta @ delta)

    best = min(distance_squared(value) for value in ordered)
    for first, second in zip(ordered, ordered[1:], strict=False):
        midpoint = (first + second) / 2.0
        point = start + direction * midpoint
        coefficients: list[tuple[float, float]] = []
        for axis in range(3):
            if point[axis] < lower[axis]:
                coefficients.append(
                    (-float(direction[axis]), float(lower[axis] - start[axis]))
                )
            elif point[axis] > upper[axis]:
                coefficients.append(
                    (float(direction[axis]), float(start[axis] - upper[axis]))
                )
        denominator = sum(slope * slope for slope, _intercept in coefficients)
        if denominator <= 0.0:
            return 0.0
        optimum = (
            -sum(slope * intercept for slope, intercept in coefficients) / denominator
        )
        if first <= optimum <= second:
            best = min(best, distance_squared(optimum))
    return best


def _build_support_index(
    primitives: Sequence[_SupportPrimitive],
    *,
    policy: TrajectorySupportPolicy,
) -> Mapping[Cell, tuple[int, ...]]:
    cell_size = policy.support_radius_m
    mutable: dict[Cell, list[int]] = defaultdict(list)
    for index, primitive in enumerate(primitives):
        start = np.asarray(primitive.start_m, dtype=np.float64)
        end = np.asarray(primitive.end_m, dtype=np.float64)
        lower = np.floor(
            (np.minimum(start, end) - primitive.radius_m) / cell_size
        ).astype(np.int64)
        upper = np.floor(
            (np.maximum(start, end) + primitive.radius_m) / cell_size
        ).astype(np.int64)
        cell_count = int(np.prod(upper - lower + 1, dtype=np.int64))
        if cell_count > policy.maximum_capsule_index_cells:
            raise TrajectorySupportError(
                TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE,
                "one support capsule exceeds maximum_capsule_index_cells",
            )
        for x in range(int(lower[0]), int(upper[0]) + 1):
            for y in range(int(lower[1]), int(upper[1]) + 1):
                for z in range(int(lower[2]), int(upper[2]) + 1):
                    mutable[(x, y, z)].append(index)
        if len(mutable) > policy.maximum_capsule_index_cells:
            raise TrajectorySupportError(
                TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE,
                "support index exceeds maximum_capsule_index_cells",
            )
    return {cell: tuple(indices) for cell, indices in mutable.items()}


def _build_occupancy_index(
    occupied: frozenset[Cell], *, policy: TrajectorySupportPolicy
) -> tuple[NDArray[np.float64], cKDTree]:
    """Build a deterministic exact-nearest index over inflated voxel centres."""
    voxel = policy.occupancy_voxel_size_m
    centers = (np.asarray(sorted(occupied), dtype=np.float64) + 0.5) * voxel
    index = cKDTree(
        centers,
        compact_nodes=True,
        balanced_tree=True,
        copy_data=True,
    )
    centers.setflags(write=False)
    return centers, index


def _readonly_centers(
    ordered: Sequence[tuple[SceneCamera, NDArray[np.float64]]],
) -> NDArray[np.float64]:
    centers = np.asarray([center for _camera, center in ordered], dtype=np.float64)
    centers.setflags(write=False)
    return centers


def _minimum_box_clearance(
    point: NDArray[np.float64],
    *,
    centers: NDArray[np.float64],
    half_extent: float,
) -> float:
    delta = np.maximum(np.abs(centers - point) - half_extent, 0.0)
    distances_squared = np.sum(delta * delta, axis=1)
    return math.sqrt(float(np.min(distances_squared)))


def _support_input_digest(
    *,
    ordered: Sequence[tuple[SceneCamera, NDArray[np.float64]]],
    points: NDArray[np.float64],
    policy: TrajectorySupportPolicy,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for camera, center in ordered:
        digest.update(camera.camera_id.encode("utf-8"))
        digest.update(str(camera.source_frame_index).encode("ascii"))
        digest.update(np.asarray(center, dtype="<f8").tobytes())
    digest.update(np.asarray(points, dtype="<f8").tobytes())
    return digest.hexdigest()


def _cell(point: NDArray[np.float64], cell_size: float) -> Cell:
    values = np.floor(point / cell_size).astype(np.int64)
    return int(values[0]), int(values[1]), int(values[2])


def _point_segment_distance(
    point: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> float:
    delta = end - start
    squared = float(delta @ delta)
    if squared <= 1.0e-24:
        return float(np.linalg.norm(point - start))
    parameter = float(np.clip(((point - start) @ delta) / squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + parameter * delta)))


__all__ = [
    "TrajectorySupportError",
    "TrajectorySupportModel",
    "build_trajectory_support_model",
    "canonical_public_camera_inventory",
    "evaluate_trajectory_safety",
    "public_camera_inventory_digest",
]
