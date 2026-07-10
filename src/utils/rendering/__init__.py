"""Rendering utilities for tennis scene visualization.

This module provides reusable rendering components for:
- Tennis court visualization (2D and 3D)
- Human skeleton rendering
- Ball trajectory visualization
- Triangle-mesh rendering (camera-view overlay and matplotlib 3D)
- 3D drawing effects (fading trails, ground shadows, impact rings)
- Trajectory kinematics (per-frame speed, bounce detection)

Example:
    >>> from src.utils.rendering import CourtRenderer, BallRenderer
    >>> court = CourtRenderer()
    >>> ball = BallRenderer()
    >>> fig, ax = plt.subplots()
    >>> court.render_2d(ax)
    >>> ball.render_trajectory_2d(ax, positions)

"""

from src.utils.rendering.ball_renderer import BallRenderer
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.effects import (
    render_fading_line_3d,
    render_ground_ring,
    render_ground_shadow,
)
from src.utils.rendering.mesh_renderer import MeshRenderer, MeshStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer
from src.utils.rendering.trajectory_analysis import compute_speeds, detect_bounces

__all__ = [
    # Renderers
    "CourtRenderer",
    "SkeletonRenderer",
    "BallRenderer",
    "MeshRenderer",
    "MeshStyle",
    # Effects
    "render_fading_line_3d",
    "render_ground_ring",
    "render_ground_shadow",
    # Trajectory kinematics
    "compute_speeds",
    "detect_bounces",
]
