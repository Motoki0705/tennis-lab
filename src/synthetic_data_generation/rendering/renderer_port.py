"""Renderer-independent contracts for captured cameras and scene-space spheres."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import SceneCamera

RENDER_RESULT_SCHEMA = "sphere_renderer_result_v1"
CAMERA_Z_DEPTH = "opencv_camera_z_scene_units"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class VisibilityState(StrEnum):
    """Mutually exclusive projected-sphere visibility state."""

    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    OUT_OF_FRAME = "out_of_frame"


@dataclass(frozen=True)
class SpherePrimitive:
    """One sphere already transformed into provider scene coordinates."""

    primitive_id: str
    center_scene: tuple[float, float, float]
    radius_scene_units: float
    color_rgb: tuple[int, int, int] = (64, 192, 64)

    def __post_init__(self) -> None:
        if _ID_PATTERN.fullmatch(self.primitive_id) is None:
            raise ValueError("Sphere primitive_id must be path-safe.")
        center = np.asarray(self.center_scene, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Sphere center_scene must contain three finite values.")
        radius = float(self.radius_scene_units)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Sphere radius_scene_units must be finite and positive.")
        color = tuple(self.color_rgb)
        if len(color) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in color
        ):
            raise ValueError("Sphere color_rgb must contain three uint8 values.")
        object.__setattr__(
            self,
            "center_scene",
            tuple(float(value) for value in center),
        )
        object.__setattr__(self, "radius_scene_units", radius)
        object.__setattr__(self, "color_rgb", color)


@dataclass(frozen=True)
class RenderRequest:
    """One fixed-camera frame request for one or more sphere primitives."""

    scene_fingerprint: str
    frame_index: int
    camera: SceneCamera
    spheres: tuple[SpherePrimitive, ...]
    supersampling: int = 4

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.scene_fingerprint) is None:
            raise ValueError("Render scene_fingerprint must be a SHA-256 digest.")
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("Render frame_index must be non-negative.")
        if (
            isinstance(self.supersampling, bool)
            or not isinstance(self.supersampling, int)
            or not 1 <= self.supersampling <= 16
        ):
            raise ValueError("Render supersampling must be an integer in [1, 16].")
        spheres = tuple(self.spheres)
        sphere_ids = [sphere.primitive_id for sphere in spheres]
        if len(sphere_ids) != len(set(sphere_ids)):
            raise ValueError("Render sphere primitive ids must be unique.")
        object.__setattr__(self, "spheres", spheres)


@dataclass(frozen=True)
class SphereRenderEvidence:
    """Geometry and visibility derived from the rendered sphere samples."""

    primitive_id: str
    projected_center_xy: tuple[float, float] | None
    apparent_diameter_px: float
    centre_depth_scene_units: float | None
    in_frame: bool
    covered_pixel_equivalent: float
    visible_pixel_equivalent: float
    visible_pixel_fraction: float
    visibility: VisibilityState

    def __post_init__(self) -> None:
        if _ID_PATTERN.fullmatch(self.primitive_id) is None:
            raise ValueError("Evidence primitive_id must be path-safe.")
        if self.projected_center_xy is not None:
            center = np.asarray(self.projected_center_xy, dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError("Projected center must contain two finite values.")
        for name, value in (
            ("apparent_diameter_px", self.apparent_diameter_px),
            ("covered_pixel_equivalent", self.covered_pixel_equivalent),
            ("visible_pixel_equivalent", self.visible_pixel_equivalent),
            ("visible_pixel_fraction", self.visible_pixel_fraction),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Evidence {name} must be finite and non-negative.")
        if self.centre_depth_scene_units is not None and (
            not np.isfinite(self.centre_depth_scene_units)
            or self.centre_depth_scene_units <= 0.0
        ):
            raise ValueError("Evidence centre depth must be finite and positive.")
        if self.visible_pixel_fraction > 1.0 + 1.0e-9:
            raise ValueError("visible_pixel_fraction must not exceed one.")
        if self.visible_pixel_equivalent > self.covered_pixel_equivalent + 1.0e-9:
            raise ValueError("Visible pixels must not exceed covered pixels.")
        if self.in_frame != (self.covered_pixel_equivalent > 0.0):
            raise ValueError("Evidence in_frame must agree with rendered coverage.")
        expected = _visibility_state(
            covered=self.covered_pixel_equivalent,
            visible=self.visible_pixel_equivalent,
        )
        if self.visibility != expected:
            raise ValueError(
                f"Evidence visibility {self.visibility!r} disagrees with {expected!r}."
            )


@dataclass(frozen=True)
class RenderMetadata:
    """Backend identity and coordinate conventions for one render result."""

    schema: str
    backend_id: str
    backend_version: str
    scene_fingerprint: str
    camera_id: str
    frame_index: int
    width: int
    height: int
    supersampling: int
    depth_convention: str
    deterministic: bool

    def __post_init__(self) -> None:
        if self.schema != RENDER_RESULT_SCHEMA:
            raise ValueError(f"Unsupported render result schema: {self.schema!r}.")
        for name, value in (
            ("backend_id", self.backend_id),
            ("backend_version", self.backend_version),
            ("camera_id", self.camera_id),
        ):
            if not value.strip():
                raise ValueError(f"Render metadata {name} must not be empty.")
        if _SHA256_PATTERN.fullmatch(self.scene_fingerprint) is None:
            raise ValueError("Render metadata scene fingerprint is invalid.")
        if self.frame_index < 0 or self.width <= 1 or self.height <= 1:
            raise ValueError("Render metadata dimensions/frame index are invalid.")
        if not 1 <= self.supersampling <= 16:
            raise ValueError("Render metadata supersampling is invalid.")
        if self.depth_convention != CAMERA_Z_DEPTH:
            raise ValueError(
                f"Unsupported render depth convention: {self.depth_convention!r}."
            )


@dataclass(frozen=True)
class RenderResult:
    """RGB and label evidence produced from one shared render calculation."""

    rgb: NDArray[np.uint8]
    scene_depth: NDArray[np.float32]
    sphere_depth: NDArray[np.float32]
    alpha: NDArray[np.float32]
    coverage: NDArray[np.float32]
    spheres: tuple[SphereRenderEvidence, ...]
    metadata: RenderMetadata

    def __post_init__(self) -> None:
        height, width = self.metadata.height, self.metadata.width
        expected_2d = (height, width)
        arrays: tuple[
            tuple[str, NDArray[Any], tuple[int, ...], np.dtype[Any]],
            ...,
        ] = (
            ("rgb", self.rgb, (height, width, 3), np.dtype(np.uint8)),
            ("scene_depth", self.scene_depth, expected_2d, np.dtype(np.float32)),
            ("sphere_depth", self.sphere_depth, expected_2d, np.dtype(np.float32)),
            ("alpha", self.alpha, expected_2d, np.dtype(np.float32)),
            ("coverage", self.coverage, expected_2d, np.dtype(np.float32)),
        )
        for name, value, shape, dtype in arrays:
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"Render {name} must have shape {shape} and dtype {dtype}."
                )
        if np.isnan(self.scene_depth).any() or np.isnan(self.sphere_depth).any():
            raise ValueError("Render depth arrays must not contain NaN.")
        if bool(np.any(self.scene_depth <= 0.0)):
            raise ValueError("Scene depth must be positive or infinity.")
        for name, value in (("alpha", self.alpha), ("coverage", self.coverage)):
            if not np.isfinite(value).all() or bool(
                np.any((value < 0.0) | (value > 1.0))
            ):
                raise ValueError(f"Render {name} must lie in [0, 1].")
        if bool(np.any(self.alpha > self.coverage + 1.0e-6)):
            raise ValueError("Render alpha must not exceed geometric coverage.")
        spheres = tuple(self.spheres)
        ids = [sphere.primitive_id for sphere in spheres]
        if len(ids) != len(set(ids)):
            raise ValueError("Render sphere evidence ids must be unique.")
        for _, value, _, _ in arrays:
            value.setflags(write=False)
        object.__setattr__(self, "spheres", spheres)


@runtime_checkable
class RendererPort(Protocol):
    """Narrow renderer boundary implemented by CPU fakes and 3DGS adapters."""

    def render(self, request: RenderRequest) -> RenderResult:
        """Render one fixed-camera frame with sphere and occlusion evidence."""
        ...


def _visibility_state(*, covered: float, visible: float) -> VisibilityState:
    if covered <= 0.0:
        return VisibilityState.OUT_OF_FRAME
    if visible <= 0.0:
        return VisibilityState.FULLY_OCCLUDED
    if visible >= covered - 1.0e-9:
        return VisibilityState.FULLY_VISIBLE
    return VisibilityState.PARTIALLY_OCCLUDED
