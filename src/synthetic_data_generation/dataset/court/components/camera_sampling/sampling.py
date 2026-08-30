"""Uniform three-dimensional arc-length sampling for closed Court orbits."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_sampling.path_geometry import (
    closed_path_points_local,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitPathSamples,
    OrbitSamplingMode,
    OrbitSamplingPolicy,
    OrbitTrajectorySpec,
    OrbitTrajectorySpecV4,
)

_DENSE_SAMPLE_COUNT = 4_096


def sample_uniform_arc_length(
    trajectory: OrbitTrajectorySpec,
    center: OrbitCenter,
    policy: OrbitSamplingPolicy,
) -> OrbitPathSamples:
    """Sample a smooth closed 3-D curve with a strict maximum adjacent step."""
    if policy.mode is not OrbitSamplingMode.UNIFORM_ARC_LENGTH:
        raise ValueError(f"Unknown orbit sampling mode: {policy.mode!r}.")
    if center.key() != (
        trajectory.center_kind,
        trajectory.center_court_instance_id,
    ):
        raise ValueError("Trajectory and resolved orbit centre disagree.")
    dense_theta = np.linspace(
        0.0,
        2.0 * math.pi,
        num=_DENSE_SAMPLE_COUNT + 1,
        endpoint=True,
        dtype=np.float64,
    )
    dense_points = _points_local(trajectory, dense_theta)
    dense_steps = np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
    if np.any(dense_steps <= 0.0) or not np.isfinite(dense_steps).all():
        raise ValueError("Trajectory does not define a finite positive-length curve.")
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(dense_steps)))
    total_length = float(cumulative[-1])
    count = max(
        policy.minimum_sample_count,
        math.ceil(total_length / policy.max_arc_step_m),
    )
    count = _round_up(count, policy.sample_count_multiple)
    while True:
        target_lengths = np.arange(count, dtype=np.float64) * (total_length / count)
        theta = np.interp(target_lengths, cumulative, dense_theta)
        local = _points_local(trajectory, theta)
        scene = center.scene_from_center.apply(local)
        closed = np.vstack((scene, scene[:1]))
        adjacent_steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        maximum = float(np.max(adjacent_steps))
        if maximum <= policy.max_arc_step_m + 1.0e-9:
            break
        count += policy.sample_count_multiple
    return OrbitPathSamples(
        trajectory_group_id=trajectory.trajectory_group_id,
        theta_radians=theta,
        points_local_m=local,
        points_scene_m=scene,
        adjacent_steps_m=adjacent_steps,
        total_arc_length_m=total_length,
    )


def _points_local(
    trajectory: OrbitTrajectorySpec,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    if isinstance(trajectory, OrbitTrajectorySpecV4):
        fractions = np.mod(theta, 2.0 * math.pi) / (2.0 * math.pi)
        fractions = np.where(np.isclose(theta, 2.0 * math.pi), 1.0, fractions)
        return closed_path_points_local(trajectory, fractions)
    major = trajectory.radius_x_m
    minor = trajectory.radius_y_m
    unrotated = np.stack(
        (major * np.cos(theta), minor * np.sin(theta)),
        axis=1,
    )
    cosine = math.cos(trajectory.orientation_radians)
    sine = math.sin(trajectory.orientation_radians)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    xy = unrotated @ rotation.T
    if trajectory.vertical_cycles == 0:
        vertical = np.full_like(theta, trajectory.base_height_m)
    else:
        vertical = trajectory.base_height_m + trajectory.vertical_amplitude_m * np.sin(
            trajectory.vertical_cycles * theta + trajectory.vertical_phase_radians
        )
    points = np.column_stack((xy, vertical))
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise ValueError(
            "Trajectory produces a non-finite or non-positive camera height."
        )
    return points


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


__all__ = ["sample_uniform_arc_length"]
