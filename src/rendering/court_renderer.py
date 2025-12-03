"""Tennis court renderer for 2D and 3D visualization.

This module re-exports from src.utils.rendering.court_renderer for backward compatibility.

Note:
    New code should import from src.utils.rendering directly.

"""

from src.utils.rendering.court_renderer import (
    CourtLines,
    CourtRenderer,
    CourtStyle,
)

__all__ = ["CourtRenderer", "CourtStyle", "CourtLines"]
