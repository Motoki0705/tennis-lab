"""Rendering utilities for tennis scene visualization.

This module provides reusable rendering components for:
- Tennis court visualization (2D and 3D)
- Human skeleton rendering
- Ball trajectory visualization
- Triangle-mesh rendering (camera-view overlay and matplotlib 3D)

Example:
    >>> from src.utils.rendering import CourtRenderer, BallRenderer
    >>> court = CourtRenderer()
    >>> ball = BallRenderer()
    >>> fig, ax = plt.subplots()
    >>> court.render_2d(ax)
    >>> ball.render_trajectory_2d(ax, positions)

"""

from src.utils.rendering.ball_renderer import BallRenderer
from src.utils.rendering.camera_view import CameraView3DConfig, ResolvedCameraView3D
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.mesh_renderer import MeshRenderer, MeshStyle
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

__all__ = [
    # Renderers
    "CourtRenderer",
    "SkeletonRenderer",
    "BallRenderer",
    "MeshRenderer",
    "MeshStyle",
    "CameraView3DConfig",
    "ResolvedCameraView3D",
]
