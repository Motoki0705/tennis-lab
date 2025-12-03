"""Rendering utilities for tennis scene visualization.

This module provides reusable rendering components for:
- Tennis court visualization (2D and 3D)
- Human skeleton rendering
- Ball trajectory visualization
- Complete scene composition

Example:
    >>> from src.utils.rendering import CourtRenderer, BallRenderer
    >>> court = CourtRenderer()
    >>> ball = BallRenderer()
    >>> fig, ax = plt.subplots()
    >>> court.render_2d(ax)
    >>> ball.render_trajectory_2d(ax, positions)

"""

from src.utils.rendering.ball_renderer import BallRenderer
from src.utils.rendering.blcs_scene_renderer import BLCSSceneRenderer
from src.utils.rendering.constants import (
    DEFAULT_BALL_COLOR,
    DEFAULT_COURT_COLOR,
    DEFAULT_FENCE_MARGIN,
    DEFAULT_LINE_COLOR,
    DEFAULT_NET_COLOR,
)
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.rendering.plcs_scene_renderer import PLCSSceneRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

__all__ = [
    # Style constants
    "DEFAULT_COURT_COLOR",
    "DEFAULT_LINE_COLOR",
    "DEFAULT_NET_COLOR",
    "DEFAULT_BALL_COLOR",
    "DEFAULT_FENCE_MARGIN",
    # Renderers
    "CourtRenderer",
    "SkeletonRenderer",
    "BallRenderer",
    # Scene renderers
    "PLCSSceneRenderer",
    "BLCSSceneRenderer",
]
