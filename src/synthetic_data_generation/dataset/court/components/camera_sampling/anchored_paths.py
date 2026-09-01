"""Deterministic public-camera-anchored rounded rectangles for Court V4."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np

from src.synthetic_data_generation.configuration import CourtTrajectoryPolicyV4
from src.synthetic_data_generation.dataset.court.components.camera_sampling.path_geometry import (
    rounded_rectangle_points_local,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportModel,
    canonical_public_camera_inventory,
    public_camera_inventory_digest,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    AnchoredRectangleProvenance,
    OrbitCenter,
    OrbitCenterKind,
    OrbitTrajectorySpecV4,
    PathConstructorV4,
    PathFamilyV4,
    VerticalProfileV4,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


def generate_anchored_rounded_rectangle_candidates(
    *,
    support_model: TrajectorySupportModel,
    cameras: Sequence[SceneCamera],
    centers: Sequence[OrbitCenter],
    policy: CourtTrajectoryPolicyV4,
    seed: int,
) -> tuple[OrbitTrajectorySpecV4, ...]:
    """Generate planar and genuinely raised rectangles at every public anchor."""
    camera_tuple = canonical_public_camera_inventory(cameras)
    center_tuple = tuple(centers)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Anchored path seed must be a non-negative integer.")
    if not camera_tuple or not center_tuple:
        raise ValueError("Anchored path generation requires cameras and orbit centres.")
    camera_centers = np.stack(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in camera_tuple]
    )
    digest = public_camera_inventory_digest(camera_tuple)
    if (
        camera_tuple != support_model.captured_cameras
        or digest != support_model.captured_camera_inventory_digest
        or camera_centers.shape != support_model.captured_camera_centers_m.shape
        or not np.allclose(
            camera_centers,
            support_model.captured_camera_centers_m,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "Anchored path camera inventory disagrees with public support authority."
        )
    complex_centers = tuple(
        center for center in center_tuple if center.center_kind is OrbitCenterKind.COMPLEX
    )
    if len(complex_centers) != 1:
        raise ValueError("Anchored paths require one exact complex orbit frame.")
    center = complex_centers[0]
    local_centers = center.scene_from_center.inverse().apply(camera_centers)
    if np.any(local_centers[:, 2] <= 0.0):
        raise ValueError("Public anchor cameras must lie above the Court plane.")
    anchor_order = list(range(len(camera_tuple)))
    random.Random(seed).shuffle(anchor_order)
    orientations = tuple(math.radians(value) for value in policy.orientations_degrees)
    result: list[OrbitTrajectorySpecV4] = []
    for anchor_rank, camera_index in enumerate(anchor_order):
        camera = camera_tuple[camera_index]
        anchor_local = local_centers[camera_index]
        anchor_scene = camera_centers[camera_index]
        orientation = orientations[(anchor_rank + seed) % len(orientations)]
        for profile in (
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.RAISED_PHASES,
        ):
            lift = (
                0.0
                if profile is VerticalProfileV4.PLANAR
                else policy.anchored_raised_lift_m
            )
            offsets = (0.0,) if lift == 0.0 else (0.0, lift, lift, 0.0)
            fractions = np.arange(
                policy.anchored_reference_point_count, dtype=np.float64
            ) / policy.anchored_reference_point_count
            points = rounded_rectangle_points_local(
                center_local_m=(
                    float(anchor_local[0]),
                    float(anchor_local[1]),
                    float(anchor_local[2]),
                ),
                half_width_m=policy.anchored_half_width_m,
                half_height_m=policy.anchored_half_height_m,
                corner_radius_m=policy.anchored_corner_radius_m,
                orientation_radians=orientation,
                vertical_profile=profile,
                vertical_phase_offsets_m=offsets,
                fractions=fractions,
            )
            provenance = AnchoredRectangleProvenance(
                camera_inventory_digest=digest,
                camera_inventory_count=len(camera_tuple),
                ordered_camera_index=camera_index,
                camera_id=camera.camera_id,
                source_frame_index=camera.source_frame_index,
                anchor_center_scene_m=(
                    float(anchor_scene[0]),
                    float(anchor_scene[1]),
                    float(anchor_scene[2]),
                ),
                anchor_center_local_m=(
                    float(anchor_local[0]),
                    float(anchor_local[1]),
                    float(anchor_local[2]),
                ),
                half_width_m=policy.anchored_half_width_m,
                half_height_m=policy.anchored_half_height_m,
                corner_radius_m=policy.anchored_corner_radius_m,
                orientation_radians=orientation,
                vertical_profile=profile,
                lift_m=lift,
                reference_points_local_m=tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in points
                ),
            )
            result.append(
                OrbitTrajectorySpecV4(
                    trajectory_id="pending",
                    trajectory_group_id="pending",
                    shape=PathFamilyV4.ROUNDED_RECTANGLE,
                    center_kind=OrbitCenterKind.COMPLEX,
                    center_court_instance_id=None,
                    base_radius_m=policy.anchored_half_width_m,
                    radius_scale=1.0,
                    axis_ratio=(
                        policy.anchored_half_height_m
                        / policy.anchored_half_width_m
                    ),
                    orientation_radians=orientation,
                    base_height_m=float(anchor_local[2]),
                    vertical_amplitude_m=lift,
                    vertical_cycles=0,
                    vertical_phase_radians=0.0,
                    curve_mode=profile,
                    constructor=PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE,
                    corner_radius_ratio=(
                        policy.anchored_corner_radius_m
                        / min(
                            policy.anchored_half_width_m,
                            policy.anchored_half_height_m,
                        )
                    ),
                    vertical_phase_offsets_m=offsets,
                    anchor_provenance=provenance,
                )
            )
    if len(result) != 2 * len(camera_tuple) or len(
        {item.semantic_key() for item in result}
    ) != len(result):
        raise ValueError(
            "Anchored generation must produce two unique paths per public camera."
        )
    return tuple(result)


def validate_anchored_trajectory_provenance(
    trajectory: OrbitTrajectorySpecV4,
    *,
    center: OrbitCenter,
    support_model: TrajectorySupportModel,
) -> None:
    """Bind one anchored path to the complete public-camera support authority."""
    if trajectory.constructor is not PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE:
        raise ValueError("Anchor provenance validation requires an anchored path.")
    provenance = trajectory.anchor_provenance
    if provenance is None:  # pragma: no cover - strict trajectory contract excludes this
        raise ValueError("Anchored path is missing typed provenance.")
    cameras = support_model.captured_cameras
    digest = public_camera_inventory_digest(cameras)
    if (
        support_model.captured_camera_inventory_digest != digest
        or provenance.camera_inventory_digest != digest
        or provenance.camera_inventory_count != len(cameras)
    ):
        raise ValueError(
            "Anchored path camera inventory digest/count disagrees with support authority."
        )
    camera = cameras[provenance.ordered_camera_index]
    camera_center_scene = camera.camera_to_scene.matrix()[:3, 3]
    camera_center_local = center.scene_from_center.inverse().apply(
        camera_center_scene
    )
    persisted_scene = np.asarray(provenance.anchor_center_scene_m, dtype=np.float64)
    persisted_local = np.asarray(provenance.anchor_center_local_m, dtype=np.float64)
    scene_from_persisted_local = center.scene_from_center.apply(persisted_local)
    if (
        provenance.camera_id != camera.camera_id
        or provenance.source_frame_index != camera.source_frame_index
        or not np.allclose(
            persisted_scene,
            camera_center_scene,
            atol=1.0e-12,
            rtol=0.0,
        )
        or not np.allclose(
            persisted_local,
            camera_center_local,
            atol=1.0e-12,
            rtol=0.0,
        )
        or not np.allclose(
            scene_from_persisted_local,
            persisted_scene,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "Anchored path camera identity/centre disagrees with support authority."
        )
    fractions = np.arange(
        len(provenance.reference_points_local_m), dtype=np.float64
    ) / len(provenance.reference_points_local_m)
    expected_reference = rounded_rectangle_points_local(
        center_local_m=provenance.anchor_center_local_m,
        half_width_m=provenance.half_width_m,
        half_height_m=provenance.half_height_m,
        corner_radius_m=provenance.corner_radius_m,
        orientation_radians=provenance.orientation_radians,
        vertical_profile=provenance.vertical_profile,
        vertical_phase_offsets_m=trajectory.vertical_phase_offsets_m,
        fractions=fractions,
    )
    if not np.allclose(
        expected_reference,
        np.asarray(provenance.reference_points_local_m, dtype=np.float64),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Anchored rounded-rectangle reference points disagree with geometry."
        )


__all__ = [
    "generate_anchored_rounded_rectangle_candidates",
    "public_camera_inventory_digest",
    "validate_anchored_trajectory_provenance",
]
