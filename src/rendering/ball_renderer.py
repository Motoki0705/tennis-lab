"""Ball trajectory renderer for 2D and 3D visualization.

This module re-exports from src.utils.rendering.ball_renderer for backward compatibility.

Note:
    New code should import from src.utils.rendering directly.

"""

from src.utils.rendering.ball_renderer import (
    EVENT_STYLES,
    BallEvent,
    BallEventType,
    BallRenderer,
    BallStyle,
)

__all__ = ["BallRenderer", "BallStyle", "BallEvent", "BallEventType", "EVENT_STYLES"]
