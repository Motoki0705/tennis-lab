"""Sample inward-looking circle and ellipse families around an SfM envelope."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.synthetic_data_generation.dataset.court.artifacts.layout import (
    MultiCourtLayout,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    MultiCourtProjection,
    project_multi_court,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True)
class OrbitFamilySpec:
    """One smooth orbit family in the reference court frame."""

    family_id: str
    shape: str
    radius_x_m: float
    radius_y_m: float
    height_m: float
    target_court_instance_id: str | None
    phase_radians: float
    sample_count: int

    def __post_init__(self) -> None:
        if self.shape not in {"circle", "ellipse"}:
            raise ValueError("Orbit shape must be circle or ellipse.")
        if self.shape == "circle" and not np.isclose(
            self.radius_x_m,
            self.radius_y_m,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError("Circle radii must be equal.")
        for name, value in (
            ("radius_x_m", self.radius_x_m),
            ("radius_y_m", self.radius_y_m),
            ("height_m", self.height_m),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if isinstance(self.sample_count, bool) or self.sample_count < 8:
            raise ValueError("sample_count must be an integer of at least eight.")


@dataclass(frozen=True)
class OrbitFrame:
    """One orbit camera with multi-court projection and support metrics."""

    family_id: str
    frame_index: int
    camera: SceneCamera
    projection: MultiCourtProjection
    nearest_captured_translation_m: float
    nearest_captured_rotation_deg: float
    collision_clearance_m: float


@dataclass(frozen=True)
class OrbitSamplingResult:
    """Accepted smooth frames and explicit basic-geometry rejections."""

    seed: int
    captured_radius_x_m: float
    captured_radius_y_m: float
    complex_center_reference_m: tuple[float, float, float]
    families: tuple[OrbitFamilySpec, ...]
    proposal_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    frames: tuple[OrbitFrame, ...]


def derive_orbit_families(
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    *,
    seed: int,
    samples_per_orbit: int = 48,
) -> tuple[OrbitFamilySpec, ...]:
    """Derive bold nested families from robust SfM camera extents."""
    camera_tuple = tuple(cameras)
    if not camera_tuple:
        raise ValueError("cameras must not be empty.")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be non-negative.")
    if samples_per_orbit < 8:
        raise ValueError("samples_per_orbit must be at least eight.")

    centers, _ = _captured_poses_reference(camera_tuple, layout)
    court_centers = layout.centers_in_reference()
    complex_center = np.mean(court_centers, axis=0)
    offsets = centers[:, :2] - complex_center[:2]
    radius_x = max(float(np.quantile(np.abs(offsets[:, 0]), 0.95)), 1.0)
    radius_y = max(float(np.quantile(np.abs(offsets[:, 1]), 0.95)), 1.0)
    heights: NDArray[np.float64] = np.asarray(
        np.quantile(centers[:, 2], (0.25, 0.5, 0.9)),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    result = []
    target_ids: tuple[str | None, ...] = (
        None,
        *(court.court_instance_id for court in layout.courts),
    )
    for shape in ("circle", "ellipse"):
        for scale_index, scale in enumerate((0.75, 1.0, 1.30)):
            height = float(heights[scale_index])
            if scale_index == 2:
                height += 1.5
            for target_index, target_id in enumerate(target_ids):
                if shape == "circle":
                    radius = max(radius_x, radius_y) * scale
                    axis_x = radius
                    axis_y = radius
                else:
                    axis_x = radius_x * scale
                    axis_y = radius_y * scale
                target_name = target_id or "complex"
                result.append(
                    OrbitFamilySpec(
                        family_id=(f"{shape}-scale-{scale:.2f}-target-{target_name}"),
                        shape=shape,
                        radius_x_m=axis_x,
                        radius_y_m=axis_y,
                        height_m=height,
                        target_court_instance_id=target_id,
                        phase_radians=float(
                            rng.uniform(0.0, 2.0 * np.pi)
                            + target_index * np.pi / max(len(target_ids), 1)
                        ),
                        sample_count=samples_per_orbit,
                    )
                )
    return tuple(result)


def sample_orbit_families(
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
    support_points_scene: NDArray[np.floating],
    *,
    seed: int,
    samples_per_orbit: int = 48,
    min_physical_points_per_court: int = 4,
    min_semantic_classes_per_court: int = 3,
    collision_neighbor_rank: int = 8,
    min_collision_clearance_m: float = 0.25,
) -> OrbitSamplingResult:
    """Generate bold orbits and retain full/partial multi-court supervision.

    Unlike the conservative baseline, nearest captured translation and rotation
    are measurements, not rejection limits. Basic collision and useful partial
    court coverage remain hard gates before an expensive NHT visual probe.
    """
    if not 1 <= min_physical_points_per_court <= 14:
        raise ValueError("min_physical_points_per_court must lie in [1, 14].")
    if not 1 <= min_semantic_classes_per_court <= 7:
        raise ValueError("min_semantic_classes_per_court must lie in [1, 7].")
    points_scene = np.asarray(support_points_scene, dtype=np.float64)
    if (
        points_scene.ndim != 2
        or points_scene.shape[1] != 3
        or points_scene.shape[0] < collision_neighbor_rank
        or not np.isfinite(points_scene).all()
    ):
        raise ValueError("support_points_scene must be finite [N,3] support.")
    if collision_neighbor_rank < 1:
        raise ValueError("collision_neighbor_rank must be positive.")
    if min_collision_clearance_m <= 0.0:
        raise ValueError("min_collision_clearance_m must be positive.")

    camera_tuple = tuple(cameras)
    families = derive_orbit_families(
        camera_tuple,
        layout,
        seed=seed,
        samples_per_orbit=samples_per_orbit,
    )
    reference = layout.reference
    captured_centers, captured_rotations = _captured_poses_reference(
        camera_tuple,
        layout,
    )
    support_reference = reference.court_from_scene.apply(points_scene)
    collision_tree = cKDTree(support_reference)
    complex_center = np.mean(layout.centers_in_reference(), axis=0)
    by_instance = {
        court.court_instance_id: court.center_in(reference) for court in layout.courts
    }
    template = camera_tuple[0]
    frames = []
    rejections: Counter[str] = Counter()
    global_index = 0
    for family in families:
        target = (
            complex_center
            if family.target_court_instance_id is None
            else by_instance[family.target_court_instance_id]
        )
        for family_frame in range(family.sample_count):
            angle = (
                2.0 * np.pi * family_frame / family.sample_count + family.phase_radians
            )
            center = complex_center.copy()
            center[0] += family.radius_x_m * np.cos(angle)
            center[1] += family.radius_y_m * np.sin(angle)
            center[2] = family.height_m
            rotation = _look_at_opencv(center, target)
            camera_to_reference = np.eye(4, dtype=np.float64)
            camera_to_reference[:3, :3] = rotation
            camera_to_reference[:3, 3] = center
            scene_rotation_from_reference = np.asarray(
                reference.scene_from_court.rotation,
                dtype=np.float64,
            ).reshape(3, 3)
            camera_to_scene = np.eye(4, dtype=np.float64)
            camera_to_scene[:3, :3] = (
                scene_rotation_from_reference @ camera_to_reference[:3, :3]
            )
            camera_to_scene[:3, 3] = reference.scene_from_court.apply(center[None])[0]
            camera = SceneCamera(
                camera_id=f"orbit_{global_index:06d}",
                source_camera_id="synthetic-sfm-envelope-orbit",
                image_uri=f"synthetic://{family.family_id}/{family_frame:06d}",
                source_frame_index=global_index,
                group_id=global_index // family.sample_count,
                width=template.width,
                height=template.height,
                intrinsics=template.intrinsics,
                camera_to_scene=tuple(
                    float(value) for value in camera_to_scene.ravel()
                ),
            )
            global_index += 1

            collision_distances, _ = collision_tree.query(
                center,
                k=collision_neighbor_rank,
            )
            clearance = float(np.atleast_1d(collision_distances)[-1])
            if clearance < min_collision_clearance_m:
                rejections["collision"] += 1
                continue
            projection = project_multi_court(camera, layout)
            has_useful_court = any(
                court.in_frame_point_count >= min_physical_points_per_court
                and court.in_frame_class_count >= min_semantic_classes_per_court
                for court in projection.courts
            )
            if not has_useful_court:
                rejections["insufficient_partial_court_coverage"] += 1
                continue

            translation, rotation_degrees = _nearest_captured_pose(
                center,
                rotation,
                captured_centers,
                captured_rotations,
            )
            frames.append(
                OrbitFrame(
                    family_id=family.family_id,
                    frame_index=family_frame,
                    camera=camera,
                    projection=projection,
                    nearest_captured_translation_m=translation,
                    nearest_captured_rotation_deg=rotation_degrees,
                    collision_clearance_m=clearance,
                )
            )
    centers, _ = _captured_poses_reference(camera_tuple, layout)
    offsets = centers[:, :2] - complex_center[:2]
    return OrbitSamplingResult(
        seed=seed,
        captured_radius_x_m=float(np.quantile(np.abs(offsets[:, 0]), 0.95)),
        captured_radius_y_m=float(np.quantile(np.abs(offsets[:, 1]), 0.95)),
        complex_center_reference_m=(
            float(complex_center[0]),
            float(complex_center[1]),
            float(complex_center[2]),
        ),
        families=families,
        proposal_count=sum(family.sample_count for family in families),
        rejection_counts=tuple(sorted(rejections.items())),
        frames=tuple(frames),
    )


def _captured_poses_reference(
    cameras: Sequence[SceneCamera],
    layout: MultiCourtLayout,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference = layout.reference
    scene_rotation_to_reference = np.asarray(
        reference.court_from_scene.rotation,
        dtype=np.float64,
    ).reshape(3, 3)
    centers = []
    rotations = []
    for camera in cameras:
        camera_to_scene = np.asarray(
            camera.camera_to_scene,
            dtype=np.float64,
        ).reshape(4, 4)
        centers.append(
            reference.court_from_scene.apply(camera_to_scene[None, :3, 3])[0]
        )
        rotations.append(scene_rotation_to_reference @ camera_to_scene[:3, :3])
    return np.stack(centers), np.stack(rotations)


def _look_at_opencv(
    center: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    forward = target - center
    norm = float(np.linalg.norm(forward))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Camera center and target must differ.")
    forward /= norm
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right_norm = float(np.linalg.norm(right))
    if right_norm <= np.finfo(np.float64).eps:
        raise ValueError("Look-at direction is parallel to court up.")
    right /= right_norm
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def _nearest_captured_pose(
    center: NDArray[np.float64],
    rotation: NDArray[np.float64],
    captured_centers: NDArray[np.float64],
    captured_rotations: NDArray[np.float64],
) -> tuple[float, float]:
    translations = np.linalg.norm(captured_centers - center, axis=1)
    traces = np.einsum("ij,kij->k", rotation, captured_rotations)
    angles = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
    combined = np.hypot(translations, angles)
    index = int(np.argmin(combined))
    return float(translations[index]), float(angles[index])
