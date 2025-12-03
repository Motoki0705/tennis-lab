"""PLCS scene renderer for player pose visualization.

This module re-exports from src.utils.rendering.plcs_scene_renderer for backward compatibility.

Note:
    New code should import from src.utils.rendering directly.

"""

from src.utils.rendering.plcs_scene_renderer import (
    COURT_SKELETON,
    PLCSSceneRenderer,
)

__all__ = ["PLCSSceneRenderer", "COURT_SKELETON"]
