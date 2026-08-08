"""Captured-offset-relative Court orbit centres and typed trajectory inventory."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.synthetic_data_generation.configuration import CourtTrajectoryPolicy
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    OrbitCurveMode,
    OrbitShape,
    OrbitTrajectorySpec,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def derive_orbit_centers(
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
) -> tuple[OrbitCenter, ...]:
    """Derive complex and per-court radii from captured camera offsets.

    Every centre gets its own robust offset distribution.  The complex centre
    is the accepted complex-bounds midpoint projected onto the reference court
    plane; court centres are their exact accepted local origins.
    """
    captured = tuple(cameras)
    if not captured:
        raise ValueError("Captured cameras must not be empty.")
    camera_centers_scene = np.stack(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in captured]
    )
    reference = _reference_court(layout)
    bounds = np.asarray(layout.complex_bounds_scene, dtype=np.float64).reshape(2, 3)
    midpoint_scene = np.mean(bounds, axis=0)
    midpoint_reference = reference.court_from_scene.apply(midpoint_scene[None, :])[0]
    midpoint_reference[2] = 0.0
    complex_transform = reference.scene_from_court.matrix()
    complex_transform[:3, 3] = reference.scene_from_court.apply(
        midpoint_reference[None, :]
    )[0]
    complex_center = _center_from_transform(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id=reference.court_instance_id,
        scene_from_center=RigidTransform.from_matrix(complex_transform),
        camera_centers_scene=camera_centers_scene,
    )
    court_centers = tuple(
        _center_from_transform(
            center_kind=OrbitCenterKind.COURT,
            court_instance_id=court.court_instance_id,
            reference_court_instance_id=court.court_instance_id,
            scene_from_center=court.scene_from_court,
            camera_centers_scene=camera_centers_scene,
        )
        for court in sorted(layout.courts, key=lambda item: item.court_instance_id)
    )
    return (complex_center, *court_centers)


def _reference_court(layout: MultiCourtLayout) -> CourtInstance:
    if layout.primary_court_instance_id is not None:
        return layout.court(layout.primary_court_instance_id)
    return min(layout.courts, key=lambda court: court.court_instance_id)


def _center_from_transform(
    *,
    center_kind: OrbitCenterKind,
    court_instance_id: str | None,
    reference_court_instance_id: str,
    scene_from_center: RigidTransform,
    camera_centers_scene: np.ndarray,
) -> OrbitCenter:
    centers_local = scene_from_center.inverse().apply(camera_centers_scene)
    offsets = np.linalg.norm(centers_local[:, :2], axis=1)
    positive = offsets[offsets > 1.0e-6]
    if positive.size == 0:
        raise ValueError(
            f"Captured cameras do not define a positive orbit radius for {center_kind.value}."
        )
    median = float(np.quantile(positive, 0.5))
    q90 = float(np.quantile(positive, 0.9))
    return OrbitCenter(
        center_kind=center_kind,
        court_instance_id=court_instance_id,
        reference_court_instance_id=reference_court_instance_id,
        scene_from_center=scene_from_center,
        base_radius_m=q90,
        captured_offset_median_m=median,
        captured_offset_q90_m=q90,
        captured_camera_count=len(camera_centers_scene),
    )


def generate_trajectory_candidates(
    policy: CourtTrajectoryPolicy,
    centers: Sequence[OrbitCenter],
    *,
    seed: int,
    stable_field_order: Sequence[str],
) -> tuple[OrbitTrajectorySpec, ...]:
    """Generate a finite typed candidate inventory in explicit stable order."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    center_tuple = tuple(centers)
    if not center_tuple:
        raise ValueError("centers must not be empty.")
    if len({center.key() for center in center_tuple}) != len(center_tuple):
        raise ValueError("Orbit centre candidates must be unique.")
    radius_scales = tuple(
        float(value)
        for value in np.linspace(
            policy.captured_offset_scale_range[0],
            policy.captured_offset_scale_range[1],
            num=3,
        )
    )
    shape_ratios: tuple[tuple[OrbitShape, float], ...] = (
        (OrbitShape.CIRCLE, 1.0),
        *tuple(
            (OrbitShape.ELLIPSE, ratio)
            for ratio in policy.axis_ratios
            if ratio <= 0.8
        ),
    )
    vertical_profiles: list[tuple[OrbitCurveMode, float, int, float]] = []
    if OrbitCurveMode.PLANAR.value in policy.curve_modes:
        vertical_profiles.append((OrbitCurveMode.PLANAR, 0.0, 0, 0.0))
    positive_amplitudes = tuple(
        value for value in policy.vertical_modulations_m if value > 0.0
    )
    if OrbitCurveMode.SINUSOIDAL_HEIGHT.value in policy.curve_modes:
        for index, amplitude in enumerate(positive_amplitudes):
            vertical_profiles.append(
                (
                    OrbitCurveMode.SINUSOIDAL_HEIGHT,
                    amplitude,
                    1 + index % 2,
                    (seed % 8 + index) * math.pi / 4.0,
                )
            )
    if not vertical_profiles:
        raise ValueError("Trajectory policy produced no valid vertical profiles.")

    raw: list[OrbitTrajectorySpec] = []
    for center in center_tuple:
        for shape, axis_ratio in shape_ratios:
            for orientation_degrees in policy.orientations_degrees:
                for radius_scale in radius_scales:
                    for base_height_m in policy.base_heights_m:
                        for curve_mode, amplitude, cycles, phase in vertical_profiles:
                            raw.append(
                                OrbitTrajectorySpec(
                                    trajectory_id="pending",
                                    trajectory_group_id="pending",
                                    shape=shape,
                                    center_kind=center.center_kind,
                                    center_court_instance_id=center.court_instance_id,
                                    base_radius_m=center.base_radius_m,
                                    radius_scale=radius_scale,
                                    axis_ratio=axis_ratio,
                                    orientation_radians=math.radians(
                                        orientation_degrees
                                    ),
                                    base_height_m=base_height_m,
                                    vertical_amplitude_m=amplitude,
                                    vertical_cycles=cycles,
                                    vertical_phase_radians=phase,
                                    curve_mode=curve_mode,
                                )
                            )
    if len({candidate.semantic_key() for candidate in raw}) != len(raw):
        raise ValueError("Trajectory policy generated duplicate typed candidates.")
    ordered = sorted(
        raw,
        key=lambda candidate: _stable_key(candidate, stable_field_order),
    )
    return tuple(
        replace(
            candidate,
            trajectory_id=f"trajectory-{index:05d}",
            trajectory_group_id=f"group-{index:05d}",
        )
        for index, candidate in enumerate(ordered)
    )


def trajectory_field_value(
    candidate: OrbitTrajectorySpec,
    field: str,
) -> object:
    """Read one declared typed selector field without parsing an ID."""
    values: dict[str, object] = {
        "shape": candidate.shape.value,
        "center_kind": (
            candidate.center_kind.value,
            candidate.center_court_instance_id,
        ),
        "axis_ratio": candidate.axis_ratio,
        "orientation_degrees": round(
            math.degrees(candidate.orientation_radians) % 360.0,
            9,
        ),
        "base_height_m": candidate.base_height_m,
        "vertical_modulation_m": candidate.vertical_amplitude_m,
        "curve_mode": candidate.curve_mode.value,
        "target_mode": "path_only",
    }
    try:
        return values[field]
    except KeyError as error:
        raise ValueError(f"Unknown typed selector field: {field!r}.") from error


def _stable_key(
    candidate: OrbitTrajectorySpec,
    field_order: Sequence[str],
) -> tuple[str, ...]:
    values = tuple(repr(trajectory_field_value(candidate, field)) for field in field_order)
    return (*values, repr(candidate.semantic_key()))


__all__ = [
    "derive_orbit_centers",
    "generate_trajectory_candidates",
    "trajectory_field_value",
]
