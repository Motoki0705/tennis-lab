"""Rendering utilities for tennis scene visualization.

This module provides reusable rendering components for:
- Tennis court visualization (2D and 3D)
- Human skeleton rendering
- Ball trajectory visualization
- Triangle-mesh rendering (camera-view overlay and matplotlib 3D)
- 3D drawing effects (fading trails, ground shadows, impact rings)
- Trajectory kinematics (per-frame speed, bounce detection)
- Virtual 3D camera control (presets, orbit, keyframes)
- Light/dark scene themes and the shared 3D layer (zorder) policy
- Generic HUD text overlay and top-down minimap primitives

Example:
    >>> from src.utils.rendering import CourtRenderer, BallRenderer
    >>> court = CourtRenderer()
    >>> ball = BallRenderer()
    >>> fig, ax = plt.subplots()
    >>> court.render_2d(ax)
    >>> ball.render_trajectory_2d(ax, positions)

"""

from src.utils.rendering.ball_renderer import BallRenderer
from src.utils.rendering.camera_geometry import (
    camera_coverage_segments,
    camera_frustum_corners,
    camera_frustum_segments,
    camera_trajectory_points,
    camera_trajectory_segments,
    camera_view_direction_segments,
)
from src.utils.rendering.camera_view import (
    CAMERA_PRESETS,
    CameraController,
    CameraKeyframe,
    CameraView3D,
    apply_scene_camera,
    resolve_camera_view,
)
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.effects import (
    render_fading_line_3d,
    render_ground_ring,
    render_ground_shadow,
    render_impact_ring,
)
from src.utils.rendering.hud import (
    HudStyle,
    format_frame_clock,
    format_speed_kmh,
    render_hud_text,
)
from src.utils.rendering.layers import SceneLayer, enable_explicit_layering
from src.utils.rendering.mesh_renderer import MeshRenderer, MeshStyle
from src.utils.rendering.minimap import MinimapRenderer, MinimapStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer
from src.utils.rendering.theme import (
    DARK_THEME,
    LIGHT_THEME,
    SceneTheme,
    apply_axes_layout_3d,
    apply_axes_theme_3d,
    apply_figure_theme,
    resolve_theme,
)
from src.utils.rendering.trajectory_analysis import compute_speeds, detect_bounces

__all__ = [
    # Renderers
    "CourtRenderer",
    "SkeletonRenderer",
    "BallRenderer",
    "MeshRenderer",
    "MeshStyle",
    "MinimapRenderer",
    "MinimapStyle",
    # Camera
    "CAMERA_PRESETS",
    "CameraController",
    "CameraKeyframe",
    "CameraView3D",
    "apply_scene_camera",
    "resolve_camera_view",
    "camera_coverage_segments",
    "camera_frustum_corners",
    "camera_frustum_segments",
    "camera_trajectory_points",
    "camera_trajectory_segments",
    "camera_view_direction_segments",
    # Theme
    "SceneTheme",
    "LIGHT_THEME",
    "DARK_THEME",
    "resolve_theme",
    "apply_figure_theme",
    "apply_axes_layout_3d",
    "apply_axes_theme_3d",
    # Layers
    "SceneLayer",
    "enable_explicit_layering",
    # HUD
    "HudStyle",
    "render_hud_text",
    "format_frame_clock",
    "format_speed_kmh",
    # Effects
    "render_fading_line_3d",
    "render_ground_ring",
    "render_ground_shadow",
    "render_impact_ring",
    # Trajectory kinematics
    "compute_speeds",
    "detect_bounces",
]
