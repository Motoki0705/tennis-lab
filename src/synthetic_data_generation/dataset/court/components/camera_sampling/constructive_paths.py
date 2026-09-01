"""Construct closed Court V4 paths from validated public-camera free space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportModel,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitTrajectorySpecV4,
    PathConstructorV4,
    PathFamilyV4,
    VerticalProfileV4,
)

if TYPE_CHECKING:
    from src.synthetic_data_generation.configuration import CourtTrajectoryPolicyV4


@dataclass(frozen=True, slots=True)
class FreeSpaceCycle:
    """One smoothed closed control path with a constructive safety provenance."""

    start_camera_index: int
    end_camera_index: int
    closure_distance_m: float
    control_points_scene_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        points = np.asarray(self.control_points_scene_m, dtype=np.float64)
        if (
            self.start_camera_index < 0
            or self.end_camera_index <= self.start_camera_index
            or not math.isfinite(self.closure_distance_m)
            or self.closure_distance_m <= 0.0
            or points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 8
            or not np.isfinite(points).all()
        ):
            raise ValueError("Free-space cycle controls are invalid.")
        points.setflags(write=False)
        object.__setattr__(self, "control_points_scene_m", points)


def construct_free_space_cycles(
    support_model: TrajectorySupportModel,
) -> tuple[FreeSpaceCycle, ...]:
    """Close trustworthy temporal camera runs with collision-checked spatial edges."""
    centers = support_model.captured_camera_centers_m
    policy = support_model.policy
    trusted = set(support_model.trusted_temporal_links)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(steps)))
    proposals: list[tuple[float, float, int, int]] = []
    for start in range(len(centers)):
        first_end = start + policy.minimum_cycle_frame_span
        last_end = min(len(centers) - 1, start + policy.maximum_cycle_frame_span)
        for end in range(first_end, last_end + 1):
            if any((index, index + 1) not in trusted for index in range(start, end)):
                continue
            closure = float(np.linalg.norm(centers[end] - centers[start]))
            if (
                closure <= policy.boundary_epsilon_m
                or closure > policy.maximum_cycle_closure_distance_m
            ):
                continue
            arc_length = float(cumulative[end] - cumulative[start])
            if arc_length <= 4.0 * policy.maximum_cycle_closure_distance_m:
                continue
            if not support_model.segment_is_safe(centers[end], centers[start]):
                continue
            proposals.append((closure, -arc_length, start, end))
    if not proposals:
        return ()
    selected: list[tuple[int, int]] = []
    minimum_index_separation = max(2, policy.minimum_cycle_frame_span // 12)
    for _closure, _negative_arc, start, end in sorted(proposals):
        if any(
            abs(start - observed_start) < minimum_index_separation
            and abs(end - observed_end) < minimum_index_separation
            for observed_start, observed_end in selected
        ):
            continue
        selected.append((start, end))
        if len(selected) >= policy.maximum_constructive_cycle_count:
            break
    cycles: list[FreeSpaceCycle] = []
    for start, end in sorted(selected):
        raw = centers[start : end + 1]
        smoothed = smooth_closed_control_path(
            raw,
            smoothing_distance_m=policy.cycle_smoothing_distance_m,
        )
        if not all(
            support_model.segment_is_safe(first, second)
            for first, second in zip(
                smoothed,
                np.roll(smoothed, -1, axis=0),
                strict=True,
            )
        ):
            continue
        cycles.append(
            FreeSpaceCycle(
                start_camera_index=start,
                end_camera_index=end,
                closure_distance_m=float(np.linalg.norm(raw[-1] - raw[0])),
                control_points_scene_m=smoothed,
            )
        )
    return tuple(cycles)


def generate_free_space_cycle_candidates(
    *,
    support_model: TrajectorySupportModel,
    centers: tuple[OrbitCenter, ...],
    policy: CourtTrajectoryPolicyV4,
) -> tuple[OrbitTrajectorySpecV4, ...]:
    """Create typed base and spatial/vertical variants from constructive cycles."""
    cycles = construct_free_space_cycles(support_model)
    if not cycles:
        return ()
    orientations = tuple(math.radians(value) for value in policy.orientations_degrees)
    result: list[OrbitTrajectorySpecV4] = []
    for cycle_index, cycle in enumerate(cycles):
        center = centers[cycle_index % len(centers)]
        local = center.scene_from_center.inverse().apply(cycle.control_points_scene_m)
        phase_fraction = (
            policy.orientations_degrees[cycle_index % len(orientations)] / 360.0
        )
        phase_offset = round(len(local) * phase_fraction) % len(local)
        phased = np.roll(local, -phase_offset, axis=0)
        if cycle_index % 2:
            phased = phased[::-1].copy()
        orientation = orientations[cycle_index % len(orientations)]
        base_height = policy.base_heights_m[cycle_index % len(policy.base_heights_m)]
        result.append(
            _cycle_spec(
                controls=phased,
                center=center,
                orientation_radians=orientation,
                base_height_m=base_height,
                radius_scale=1.0,
            )
        )
        variant = phased.copy()
        variant_kind = cycle_index % 3
        if variant_kind == 0:
            variant[:, :2] *= 0.97
            radius_scale = 0.97
        elif variant_kind == 1:
            variant[:, 2] += 0.25
            radius_scale = 1.0
        else:
            phase = np.linspace(0.0, 2.0 * math.pi, len(variant), endpoint=False)
            variant[:, 2] += 0.25 * np.sin(phase)
            radius_scale = 1.0
        result.append(
            _cycle_spec(
                controls=variant,
                center=center,
                orientation_radians=orientation,
                base_height_m=base_height,
                radius_scale=radius_scale,
            )
        )
    return tuple(result)


def smooth_closed_control_path(
    points: NDArray[np.floating],
    *,
    smoothing_distance_m: float,
) -> NDArray[np.float64]:
    """Round every vertex locally without cutting across the captured corridor."""
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) < 8
        or not np.isfinite(values).all()
        or not math.isfinite(smoothing_distance_m)
        or smoothing_distance_m <= 0.0
    ):
        raise ValueError("Closed smoothing input is invalid.")
    smoothed: list[NDArray[np.float64]] = []
    for index, vertex in enumerate(values):
        previous = values[index - 1]
        following = values[(index + 1) % len(values)]
        previous_distance = float(np.linalg.norm(vertex - previous))
        following_distance = float(np.linalg.norm(following - vertex))
        if min(previous_distance, following_distance) <= 1.0e-9:
            raise ValueError("Closed smoothing input contains a zero-length edge.")
        trim = min(
            smoothing_distance_m,
            0.2 * previous_distance,
            0.2 * following_distance,
        )
        before = vertex + (previous - vertex) * (trim / previous_distance)
        after = vertex + (following - vertex) * (trim / following_distance)
        for parameter in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
            one_minus = 1.0 - parameter
            smoothed.append(
                one_minus * one_minus * before
                + 2.0 * one_minus * parameter * vertex
                + parameter * parameter * after
            )
    raw_result = np.asarray(smoothed, dtype=np.float64)
    keep: NDArray[np.bool_] = np.ones(len(raw_result), dtype=np.bool_)
    keep[1:] = np.linalg.norm(np.diff(raw_result, axis=0), axis=1) > 1.0e-9
    result = raw_result[keep]
    if len(result) >= 2 and float(np.linalg.norm(result[-1] - result[0])) <= 1.0e-9:
        result = result[:-1]
    if len(result) < 8 or np.any(
        np.linalg.norm(np.diff(np.vstack((result, result[:1])), axis=0), axis=1)
        <= 1.0e-9
    ):
        raise ValueError("Closed smoothing produced a zero-length edge.")
    return result


def _cycle_spec(
    *,
    controls: NDArray[np.float64],
    center: OrbitCenter,
    orientation_radians: float,
    base_height_m: float,
    radius_scale: float,
) -> OrbitTrajectorySpecV4:
    amplitude = 0.5 * float(np.ptp(controls[:, 2]))
    cosine = math.cos(orientation_radians)
    sine = math.sin(orientation_radians)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    parameters = controls.copy()
    parameters[:, :2] = controls[:, :2] @ rotation
    parameters[:, 2] -= base_height_m
    return OrbitTrajectorySpecV4(
        trajectory_id="pending",
        trajectory_group_id="pending",
        shape=PathFamilyV4.FREE_SPACE_CYCLE,
        center_kind=center.center_kind,
        center_court_instance_id=center.court_instance_id,
        base_radius_m=center.base_radius_m,
        radius_scale=radius_scale,
        axis_ratio=1.0,
        orientation_radians=orientation_radians,
        base_height_m=base_height_m,
        vertical_amplitude_m=amplitude,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=VerticalProfileV4.FREE_SPACE_CYCLE,
        constructor=PathConstructorV4.FREE_SPACE_CYCLE,
        corner_radius_ratio=None,
        vertical_phase_offsets_m=(0.0,),
        control_points_local_m=tuple(
            (float(point[0]), float(point[1]), float(point[2])) for point in parameters
        ),
    )


__all__ = [
    "FreeSpaceCycle",
    "construct_free_space_cycles",
    "generate_free_space_cycle_candidates",
    "smooth_closed_control_path",
]
