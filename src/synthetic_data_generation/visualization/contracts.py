"""Public contracts for canonical synthetic-dataset video visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias, cast

VISUALIZATION_METADATA_SCHEMA = "canonical_synthetic_dataset_visualization_v1"
VISUALIZATION_METADATA_SCHEMA_V2 = "canonical_synthetic_dataset_visualization_v2"


class DatasetVisualizationDomain(StrEnum):
    """Canonical generated-dataset domains supported by the visualizer."""

    COURT = "court"
    BLCS = "blcs"
    PLCS = "plcs"


class CourtOverlayMode(StrEnum):
    """Explicit Court visualization geometry authority."""

    SEMANTIC = "semantic"
    TRAJECTORY_SUPPORT_AABB = "trajectory_support_aabb"


class CourtAABBRenderStyle(StrEnum):
    """Explicit rasterization style for the exact Court support occupancy."""

    WIREFRAME = "wireframe"
    SOLID = "solid"


class CourtAABBWireframeTopology(StrEnum):
    """Explicit edge-selection policy for Court AABB wireframes."""

    BOUNDARY = "boundary"
    ALL_EDGES = "all_edges"


class CourtAABBTrajectoryFilterScope(StrEnum):
    """Display-only occupancy scope around the selected closed trajectory."""

    LOCAL_SWEPT_SEGMENTS = "local_swept_segments"
    SELECTED_TRAJECTORY = "selected_trajectory"
    ALL = "all"


class CourtAABBTrajectoryFilterRadiusMode(StrEnum):
    """Authority used to resolve a non-all trajectory filter radius."""

    SUPPORT_RADIUS = "support_radius"
    EXPLICIT_RADIUS = "explicit_radius"


@dataclass(frozen=True, slots=True)
class CourtOverlayConfiguration:
    """Rendering and fail-closed resource policy for Court overlays."""

    mode: CourtOverlayMode
    render_style: CourtAABBRenderStyle
    wireframe_topology: CourtAABBWireframeTopology
    trajectory_filter_scope: CourtAABBTrajectoryFilterScope
    trajectory_filter_radius_mode: CourtAABBTrajectoryFilterRadiusMode | None
    trajectory_filter_radius_m: float | None
    color_rgb: tuple[int, int, int]
    background_color_rgb: tuple[int, int, int]
    opacity: float
    edge_opacity: float
    edge_width_px: int
    depth_epsilon_m: float
    near_plane_m: float
    maximum_cells: int
    maximum_surface_faces: int
    maximum_edge_segments: int
    maximum_projected_pixels: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CourtOverlayMode):
            raise TypeError("Court overlay mode must be CourtOverlayMode.")
        if not isinstance(self.render_style, CourtAABBRenderStyle):
            raise TypeError("Court AABB render_style must be CourtAABBRenderStyle.")
        if not isinstance(self.wireframe_topology, CourtAABBWireframeTopology):
            raise TypeError(
                "Court AABB wireframe_topology must be CourtAABBWireframeTopology."
            )
        if not isinstance(self.trajectory_filter_scope, CourtAABBTrajectoryFilterScope):
            raise TypeError(
                "Court AABB trajectory_filter_scope must be "
                "CourtAABBTrajectoryFilterScope."
            )
        radius_mode = self.trajectory_filter_radius_mode
        if radius_mode is not None and not isinstance(
            radius_mode,
            CourtAABBTrajectoryFilterRadiusMode,
        ):
            raise TypeError(
                "Court AABB trajectory_filter_radius_mode must be "
                "CourtAABBTrajectoryFilterRadiusMode or None."
            )
        filter_radius: float | None
        if self.trajectory_filter_radius_m is None:
            filter_radius = None
        else:
            filter_radius = _finite_number(
                self.trajectory_filter_radius_m,
                name="Court AABB trajectory_filter_radius_m",
            )
        if self.trajectory_filter_scope is CourtAABBTrajectoryFilterScope.ALL:
            if radius_mode is not None or filter_radius is not None:
                raise ValueError(
                    "Court AABB all filter requires radius_mode and radius to be None."
                )
        elif radius_mode is None:
            raise ValueError(
                "Court AABB non-all filter requires trajectory_filter_radius_mode."
            )
        elif radius_mode is CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS:
            if filter_radius is None or filter_radius <= 0.0:
                raise ValueError(
                    "Court AABB explicit_radius filter requires a positive finite "
                    "trajectory_filter_radius_m."
                )
        elif filter_radius is not None:
            raise ValueError(
                "Court AABB support_radius filter requires "
                "trajectory_filter_radius_m=None."
            )
        color = _rgb_bytes(self.color_rgb, name="color_rgb")
        background_color = _rgb_bytes(
            self.background_color_rgb,
            name="background_color_rgb",
        )
        opacity = _finite_number(self.opacity, name="Court overlay opacity")
        if not 0.0 < opacity <= 1.0:
            raise ValueError("Court overlay opacity must lie in (0, 1].")
        edge_opacity = _finite_number(
            self.edge_opacity,
            name="Court overlay edge_opacity",
        )
        if not 0.0 < edge_opacity <= 1.0:
            raise ValueError("Court overlay edge_opacity must lie in (0, 1].")
        if (
            isinstance(self.edge_width_px, bool)
            or not isinstance(self.edge_width_px, int)
            or not 1 <= self.edge_width_px <= 64
        ):
            raise ValueError("Court overlay edge_width_px must lie in [1, 64].")
        depth_epsilon = _finite_number(
            self.depth_epsilon_m,
            name="Court overlay depth_epsilon_m",
        )
        if depth_epsilon < 0.0:
            raise ValueError("Court overlay depth_epsilon_m must be non-negative.")
        near_plane = _finite_number(
            self.near_plane_m,
            name="Court overlay near_plane_m",
        )
        if near_plane <= 0.0:
            raise ValueError("Court overlay near_plane_m must be positive.")
        for name in (
            "maximum_cells",
            "maximum_surface_faces",
            "maximum_edge_segments",
            "maximum_projected_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Court overlay {name} must be a positive integer.")
        object.__setattr__(self, "color_rgb", color)
        object.__setattr__(self, "background_color_rgb", background_color)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "edge_opacity", edge_opacity)
        object.__setattr__(self, "trajectory_filter_radius_m", filter_radius)
        object.__setattr__(self, "depth_epsilon_m", depth_epsilon)
        object.__setattr__(self, "near_plane_m", near_plane)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic sidecar/config evidence."""
        return {
            "mode": self.mode.value,
            "render_style": self.render_style.value,
            "wireframe_topology": self.wireframe_topology.value,
            "trajectory_filter_scope": self.trajectory_filter_scope.value,
            "trajectory_filter_radius_mode": (
                None
                if self.trajectory_filter_radius_mode is None
                else self.trajectory_filter_radius_mode.value
            ),
            "trajectory_filter_radius_m": self.trajectory_filter_radius_m,
            "color_rgb": list(self.color_rgb),
            "background_color_rgb": list(self.background_color_rgb),
            "opacity": self.opacity,
            "edge_opacity": self.edge_opacity,
            "edge_width_px": self.edge_width_px,
            "depth_epsilon_m": self.depth_epsilon_m,
            "near_plane_m": self.near_plane_m,
            "maximum_cells": self.maximum_cells,
            "maximum_surface_faces": self.maximum_surface_faces,
            "maximum_edge_segments": self.maximum_edge_segments,
            "maximum_projected_pixels": self.maximum_projected_pixels,
        }


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _rgb_bytes(value: object, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3 or any(
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(f"Court overlay {name} must contain three bytes.")
    return cast(tuple[int, int, int], value)


DEFAULT_COURT_OVERLAY_CONFIGURATION = CourtOverlayConfiguration(
    mode=CourtOverlayMode.SEMANTIC,
    render_style=CourtAABBRenderStyle.WIREFRAME,
    wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
    trajectory_filter_scope=CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS,
    trajectory_filter_radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
    trajectory_filter_radius_m=1.5,
    color_rgb=(255, 96, 32),
    background_color_rgb=(0, 0, 0),
    opacity=0.55,
    edge_opacity=0.40,
    edge_width_px=1,
    depth_epsilon_m=0.02,
    near_plane_m=0.05,
    maximum_cells=1_000_000,
    maximum_surface_faces=4_000_000,
    maximum_edge_segments=8_000_000,
    maximum_projected_pixels=100_000_000,
)


@dataclass(frozen=True, slots=True, init=False)
class DatasetVisualizationConfiguration:
    """One fail-closed visualization selection and encoding contract."""

    domain: DatasetVisualizationDomain
    dataset_root: Path
    output_video: Path
    trajectory_id: str | None
    logical_scene_id: str | None
    camera_id: str | None
    fps: float
    crf: int
    history_frames: int
    court_overlay: CourtOverlayConfiguration

    def __init__(
        self,
        domain: DatasetVisualizationDomain,
        dataset_root: Path,
        output_video: Path,
        trajectory_id: str | None,
        logical_scene_id: str | None,
        camera_id: str | None,
        fps: float,
        crf: int,
        history_frames: int,
        court_overlay: CourtOverlayConfiguration = (
            DEFAULT_COURT_OVERLAY_CONFIGURATION
        ),
    ) -> None:
        """Initialize with the legacy semantic default for direct callers."""
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "output_video", output_video)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "logical_scene_id", logical_scene_id)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "crf", crf)
        object.__setattr__(self, "history_frames", history_frames)
        object.__setattr__(self, "court_overlay", court_overlay)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.domain, DatasetVisualizationDomain):
            raise TypeError("domain must be a DatasetVisualizationDomain.")
        dataset_root = _absolute_path(self.dataset_root, name="dataset_root")
        output_video = _absolute_path(self.output_video, name="output_video")
        expected_owner_parts = ("datasets", self.domain.value)
        if dataset_root.parts[-2:] != expected_owner_parts:
            raise ValueError(
                "dataset_root must be the canonical published owner "
                f".../{'/'.join(expected_owner_parts)}."
            )
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise ValueError("dataset_root must be an existing ordinary directory.")
        if output_video.suffix.lower() != ".mp4":
            raise ValueError("output_video must use the .mp4 suffix.")
        if output_video.resolve(strict=False).is_relative_to(
            dataset_root.resolve(strict=True)
        ):
            raise ValueError(
                "Visualization output must stay outside the dataset owner."
            )
        metadata_path = output_video.with_suffix(".json")
        if output_video.exists() or output_video.is_symlink():
            raise FileExistsError(f"Visualization video already exists: {output_video}")
        if metadata_path.exists() or metadata_path.is_symlink():
            raise FileExistsError(
                f"Visualization metadata already exists: {metadata_path}"
            )
        trajectory_id = _optional_identifier(self.trajectory_id, name="trajectory_id")
        logical_scene_id = _optional_identifier(
            self.logical_scene_id, name="logical_scene_id"
        )
        camera_id = _optional_identifier(self.camera_id, name="camera_id")
        if not isinstance(self.court_overlay, CourtOverlayConfiguration):
            raise TypeError("court_overlay must be CourtOverlayConfiguration.")
        if self.domain is DatasetVisualizationDomain.COURT:
            if trajectory_id is None:
                raise ValueError("Court visualization requires trajectory_id.")
            if logical_scene_id is not None or camera_id is not None:
                raise ValueError(
                    "Court visualization does not accept logical_scene_id or camera_id."
                )
        elif trajectory_id is not None:
            raise ValueError("BLCS/PLCS visualization does not accept trajectory_id.")
        elif logical_scene_id is None or camera_id is None:
            raise ValueError(
                "BLCS/PLCS visualization requires logical_scene_id and camera_id."
            )
        if (
            self.domain is not DatasetVisualizationDomain.COURT
            and self.court_overlay.mode is not CourtOverlayMode.SEMANTIC
        ):
            raise ValueError(
                "trajectory_support_aabb overlay is accepted only for Court."
            )
        fps = float(self.fps)
        if not math.isfinite(fps) or not 0.0 < fps <= 240.0:
            raise ValueError("fps must be finite and lie in (0, 240].")
        if isinstance(self.crf, bool) or not isinstance(self.crf, int):
            raise TypeError("crf must be an integer.")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must lie in [0, 51].")
        if (
            isinstance(self.history_frames, bool)
            or not isinstance(self.history_frames, int)
            or not 0 <= self.history_frames <= 120
        ):
            raise ValueError("history_frames must be an integer in [0, 120].")
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "output_video", output_video)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "logical_scene_id", logical_scene_id)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "fps", fps)

    @property
    def metadata_path(self) -> Path:
        """Return the deterministic sidecar path paired with the MP4."""
        return self.output_video.with_suffix(".json")


@dataclass(frozen=True, slots=True)
class DatasetVisualizationResult:
    """Published video and deterministic metadata summary."""

    video_path: Path
    metadata_path: Path
    frame_count: int
    width: int
    height: int


def _absolute_path(value: Path, *, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path.")
    return value


def _optional_identifier(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be null or a non-empty trimmed string.")
    return value


# The public request name describes its runtime role. The defining class keeps
# the repository-wide ``*Configuration`` naming contract so source discovery
# can expose its fields without maintaining a second schema.
DatasetVisualizationRequest: TypeAlias = DatasetVisualizationConfiguration


__all__ = [
    "DatasetVisualizationDomain",
    "DatasetVisualizationConfiguration",
    "DatasetVisualizationRequest",
    "DatasetVisualizationResult",
    "DEFAULT_COURT_OVERLAY_CONFIGURATION",
    "VISUALIZATION_METADATA_SCHEMA",
    "VISUALIZATION_METADATA_SCHEMA_V2",
    "CourtOverlayConfiguration",
    "CourtOverlayMode",
    "CourtAABBRenderStyle",
    "CourtAABBWireframeTopology",
    "CourtAABBTrajectoryFilterRadiusMode",
    "CourtAABBTrajectoryFilterScope",
]
