"""Rendering utilities for tennis scene visualization.

This module provides reusable rendering components for:
- Tennis court visualization (2D and 3D)
- Human skeleton rendering
- Ball trajectory visualization

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
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

__all__ = [
    # Renderers
    "CourtRenderer",
    "SkeletonRenderer",
    "BallRenderer",
]
