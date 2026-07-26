"""Deterministic CPU reference renderer for sphere-label contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.rendering.renderer_port import (
    CAMERA_Z_DEPTH,
    RENDER_RESULT_SCHEMA,
    RenderMetadata,
    RenderRequest,
    RenderResult,
    SpherePrimitive,
    SphereRenderEvidence,
    _visibility_state,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True)
class CpuSceneFrame:
    """Static RGB/depth buffers for one captured camera."""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("CPU scene RGB must have shape (H, W, 3).")
        if self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("CPU scene depth shape must match RGB.")
        if self.rgb.dtype != np.uint8 or self.depth.dtype != np.float32:
            raise ValueError("CPU scene buffers must use uint8 RGB and float32 depth.")
        if np.isnan(self.depth).any() or bool(np.any(self.depth <= 0.0)):
            raise ValueError("CPU scene depth must be positive or infinity.")


class DeterministicCpuSphereRenderer:
    """Ray/sphere supersampling reference with explicit static-scene depth."""

    def __init__(
        self,
        *,
        scene_fingerprint: str,
        frames: dict[str, CpuSceneFrame],
        depth_epsilon: float = 1.0e-6,
    ) -> None:
        if len(scene_fingerprint) != 64:
            raise ValueError("CPU renderer scene fingerprint must be SHA-256.")
        if not frames:
            raise ValueError("CPU renderer requires at least one camera frame.")
        if depth_epsilon < 0.0 or not np.isfinite(depth_epsilon):
            raise ValueError(
                "CPU renderer depth_epsilon must be finite and non-negative."
            )
        self._scene_fingerprint = scene_fingerprint
        self._frames = dict(frames)
        self._depth_epsilon = float(depth_epsilon)

    def render(self, request: RenderRequest) -> RenderResult:
        """Render request using the same samples for image and label evidence."""
        if request.scene_fingerprint != self._scene_fingerprint:
            raise ValueError("Render request scene fingerprint mismatch.")
        try:
            frame = self._frames[request.camera.camera_id]
        except KeyError as exc:
            raise ValueError(
                f"No CPU scene frame for camera {request.camera.camera_id!r}."
            ) from exc
        height, width = frame.depth.shape
        if (request.camera.height, request.camera.width) != (height, width):
            raise ValueError("CPU scene frame dimensions differ from SceneCamera.")
        if not request.spheres:
            return _empty_result(request=request, frame=frame, spheres=())

        camera_to_scene = np.asarray(
            request.camera.camera_to_scene,
            dtype=np.float64,
        ).reshape(4, 4)
        intrinsics = np.asarray(
            request.camera.intrinsics,
            dtype=np.float64,
        ).reshape(3, 3)
        region = _conservative_render_region(
            spheres=request.spheres,
            camera_to_scene=camera_to_scene,
            intrinsics=intrinsics,
            width=width,
            height=height,
        )
        if region is None:
            sample_count = request.supersampling**2
            empty_samples: NDArray[np.bool_] = np.zeros(
                (0, 0, sample_count), dtype=np.bool_
            )
            evidence = tuple(
                _sphere_evidence(
                    sphere,
                    camera=request.camera,
                    covered=empty_samples,
                    visible=empty_samples,
                    sample_count=sample_count,
                )
                for sphere in request.spheres
            )
            return _empty_result(
                request=request,
                frame=frame,
                spheres=evidence,
            )
        x_start, x_stop, y_start, y_stop = region

        sample_u, sample_v = _subpixel_grid(
            x_start=x_start,
            x_stop=x_stop,
            y_start=y_start,
            y_stop=y_stop,
            supersampling=request.supersampling,
        )
        rays = np.stack(
            (
                (sample_u - intrinsics[0, 2]) / intrinsics[0, 0],
                (sample_v - intrinsics[1, 2]) / intrinsics[1, 1],
                np.ones_like(sample_u),
            ),
            axis=-1,
        )
        sphere_depths = [
            _sphere_depth_samples(
                rays,
                sphere=sphere,
                camera_to_scene=camera_to_scene,
            )
            for sphere in request.spheres
        ]
        sample_count = request.supersampling**2
        sample_scene_depth = np.repeat(
            frame.depth[y_start:y_stop, x_start:x_stop, None],
            sample_count,
            axis=2,
        )
        depth_stack = np.stack(sphere_depths, axis=0)
        nearest_index = np.argmin(depth_stack, axis=0)
        nearest_depth = np.min(depth_stack, axis=0)
        any_coverage = np.isfinite(nearest_depth)
        any_visible = any_coverage & (
            nearest_depth < sample_scene_depth - self._depth_epsilon
        )

        sample_rgb = np.repeat(
            frame.rgb[y_start:y_stop, x_start:x_stop, None, :],
            sample_count,
            axis=2,
        )
        evidence_records: list[SphereRenderEvidence] = []
        for sphere_index, sphere in enumerate(request.spheres):
            covered = np.isfinite(depth_stack[sphere_index])
            visible = covered & (nearest_index == sphere_index) & any_visible
            sample_rgb[visible] = np.asarray(sphere.color_rgb, dtype=np.uint8)
            evidence_records.append(
                _sphere_evidence(
                    sphere,
                    camera=request.camera,
                    covered=covered,
                    visible=visible,
                    sample_count=sample_count,
                )
            )

        rgb = frame.rgb.copy()
        rgb[y_start:y_stop, x_start:x_stop] = np.rint(sample_rgb.mean(axis=2)).astype(
            np.uint8
        )
        coverage = np.zeros((height, width), dtype=np.float32)
        coverage[y_start:y_stop, x_start:x_stop] = any_coverage.mean(
            axis=2,
            dtype=np.float64,
        ).astype(np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[y_start:y_stop, x_start:x_stop] = any_visible.mean(
            axis=2,
            dtype=np.float64,
        ).astype(np.float32)
        sphere_depth = np.full((height, width), np.inf, dtype=np.float32)
        sphere_depth[y_start:y_stop, x_start:x_stop] = np.min(
            nearest_depth,
            axis=2,
        ).astype(np.float32)
        return RenderResult(
            rgb=rgb,
            scene_depth=frame.depth.copy(),
            sphere_depth=sphere_depth,
            alpha=alpha,
            coverage=coverage,
            spheres=tuple(evidence_records),
            metadata=_metadata(request),
        )


def _empty_result(
    *,
    request: RenderRequest,
    frame: CpuSceneFrame,
    spheres: tuple[SphereRenderEvidence, ...],
) -> RenderResult:
    height, width = frame.depth.shape
    empty = np.zeros((height, width), dtype=np.float32)
    return RenderResult(
        rgb=frame.rgb.copy(),
        scene_depth=frame.depth.copy(),
        sphere_depth=np.full((height, width), np.inf, dtype=np.float32),
        alpha=empty.copy(),
        coverage=empty,
        spheres=spheres,
        metadata=_metadata(request),
    )


def _metadata(request: RenderRequest) -> RenderMetadata:
    return RenderMetadata(
        schema=RENDER_RESULT_SCHEMA,
        backend_id="deterministic-cpu-sphere-reference",
        backend_version="1",
        scene_fingerprint=request.scene_fingerprint,
        camera_id=request.camera.camera_id,
        frame_index=request.frame_index,
        width=request.camera.width,
        height=request.camera.height,
        supersampling=request.supersampling,
        depth_convention=CAMERA_Z_DEPTH,
        deterministic=True,
    )


def _subpixel_grid(
    *,
    x_start: int,
    x_stop: int,
    y_start: int,
    y_stop: int,
    supersampling: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    width = x_stop - x_start
    height = y_stop - y_start
    offsets = (np.arange(supersampling, dtype=np.float64) + 0.5) / supersampling - 0.5
    columns, rows, offset_y, offset_x = np.meshgrid(
        np.arange(x_start, x_stop, dtype=np.float64),
        np.arange(y_start, y_stop, dtype=np.float64),
        offsets,
        offsets,
        indexing="xy",
    )
    return (
        (columns + offset_x).reshape(height, width, -1),
        (rows + offset_y).reshape(height, width, -1),
    )


def _conservative_render_region(
    *,
    spheres: tuple[SpherePrimitive, ...],
    camera_to_scene: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Return a clipped ROI guaranteed to contain every projected sphere."""
    x_min = width
    x_max = 0
    y_min = height
    y_max = 0
    found = False
    rotation = camera_to_scene[:3, :3]
    translation = camera_to_scene[:3, 3]
    for sphere in spheres:
        center = rotation.T @ (
            np.asarray(sphere.center_scene, dtype=np.float64) - translation
        )
        radius = sphere.radius_scene_units
        if center[2] + radius <= 0.0:
            continue
        if center[2] <= radius:
            return (0, width, 0, height)
        projected = intrinsics[:2, :2] @ (center[:2] / center[2]) + intrinsics[:2, 2]
        denominator = center[2] * (center[2] - radius)
        half_width = (
            abs(intrinsics[0, 0]) * radius * (center[2] + abs(center[0])) / denominator
            + 2.0
        )
        half_height = (
            abs(intrinsics[1, 1]) * radius * (center[2] + abs(center[1])) / denominator
            + 2.0
        )
        sphere_x_min = max(0, int(np.floor(projected[0] - half_width)))
        sphere_x_max = min(width, int(np.ceil(projected[0] + half_width)) + 1)
        sphere_y_min = max(0, int(np.floor(projected[1] - half_height)))
        sphere_y_max = min(height, int(np.ceil(projected[1] + half_height)) + 1)
        if sphere_x_min >= sphere_x_max or sphere_y_min >= sphere_y_max:
            continue
        found = True
        x_min = min(x_min, sphere_x_min)
        x_max = max(x_max, sphere_x_max)
        y_min = min(y_min, sphere_y_min)
        y_max = max(y_max, sphere_y_max)
    if not found:
        return None
    return (x_min, x_max, y_min, y_max)


def _sphere_depth_samples(
    rays: NDArray[np.float64],
    *,
    sphere: SpherePrimitive,
    camera_to_scene: NDArray[np.float64],
) -> NDArray[np.float64]:
    rotation = camera_to_scene[:3, :3]
    translation = camera_to_scene[:3, 3]
    center_camera = rotation.T @ (
        np.asarray(sphere.center_scene, dtype=np.float64) - translation
    )
    a = np.sum(np.square(rays), axis=-1)
    b = -2.0 * np.sum(rays * center_camera, axis=-1)
    c = float(center_camera @ center_camera - sphere.radius_scene_units**2)
    discriminant = np.square(b) - 4.0 * a * c
    valid = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = (-b - root) / (2.0 * a)
    far = (-b + root) / (2.0 * a)
    depth = np.where(near > 0.0, near, np.where(far > 0.0, far, np.inf))
    return np.where(valid, depth, np.inf)


def _sphere_evidence(
    sphere: SpherePrimitive,
    *,
    camera: SceneCamera,
    covered: NDArray[np.bool_],
    visible: NDArray[np.bool_],
    sample_count: int,
) -> SphereRenderEvidence:
    camera_to_scene = np.asarray(
        camera.camera_to_scene,
        dtype=np.float64,
    ).reshape(4, 4)
    intrinsics = np.asarray(
        camera.intrinsics,
        dtype=np.float64,
    ).reshape(3, 3)
    center_camera = camera_to_scene[:3, :3].T @ (
        np.asarray(sphere.center_scene, dtype=np.float64) - camera_to_scene[:3, 3]
    )
    if center_camera[2] > 0.0:
        projected = (
            intrinsics[:2, :2] @ (center_camera[:2] / center_camera[2])
            + intrinsics[:2, 2]
        )
        projected_center = (float(projected[0]), float(projected[1]))
        focal_geometric_mean = float(np.sqrt(intrinsics[0, 0] * intrinsics[1, 1]))
        diameter = (
            2.0
            * focal_geometric_mean
            * sphere.radius_scene_units
            / float(center_camera[2])
        )
        centre_depth = float(center_camera[2])
    else:
        projected_center = None
        diameter = 0.0
        centre_depth = None
    covered_samples = int(np.count_nonzero(covered))
    visible_samples = int(np.count_nonzero(visible))
    covered_pixels = covered_samples / sample_count
    visible_pixels = visible_samples / sample_count
    visible_fraction = visible_samples / covered_samples if covered_samples else 0.0
    return SphereRenderEvidence(
        primitive_id=sphere.primitive_id,
        projected_center_xy=projected_center,
        apparent_diameter_px=diameter,
        centre_depth_scene_units=centre_depth,
        in_frame=covered_samples > 0,
        covered_pixel_equivalent=covered_pixels,
        visible_pixel_equivalent=visible_pixels,
        visible_pixel_fraction=visible_fraction,
        visibility=_visibility_state(
            covered=covered_pixels,
            visible=visible_pixels,
        ),
    )
