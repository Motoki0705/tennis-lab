"""Rendering module for tennis scene visualization.

This module provides reusable rendering components for:
- Tennis court visualization (2D and 3D)
- Human skeleton rendering
- Ball trajectory visualization
- Complete scene composition

Example:
    >>> from src.rendering import CourtRenderer, BallRenderer
    >>> court = CourtRenderer()
    >>> ball = BallRenderer()
    >>> fig, ax = plt.subplots()
    >>> court.render_2d(ax)
    >>> ball.render_trajectory_2d(ax, positions)

Note:
    This module re-exports from src.utils.rendering for backward compatibility.
    New code should import from src.utils.rendering directly.

"""

# Re-export renderers from utils.rendering
# Re-export geometry constants for backward compatibility
from src.utils.geometry.court import (
    CENTER_MARK_LENGTH,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    SERVICE_LINE_DISTANCE,
)
from src.utils.rendering import (
    BallRenderer,
    BLCSSceneRenderer,
    CourtRenderer,
    PLCSSceneRenderer,
    SkeletonRenderer,
)

__all__ = [
    # Constants (from utils.geometry.court)
    "HALF_LENGTH",
    "HALF_SINGLES_WIDTH",
    "HALF_DOUBLES_WIDTH",
    "SERVICE_LINE_DISTANCE",
    "CENTER_MARK_LENGTH",
    "NET_HEIGHT_CENTER",
    "NET_HEIGHT_POST",
    # Base renderers
    "CourtRenderer",
    "SkeletonRenderer",
    "BallRenderer",
    # Scene renderers
    "PLCSSceneRenderer",
    "BLCSSceneRenderer",
]
