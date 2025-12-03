"""BLCS scene renderer for ball trajectory visualization.

This module re-exports from src.utils.rendering.blcs_scene_renderer for backward compatibility.

Note:
    New code should import from src.utils.rendering directly.

"""

from src.utils.rendering.blcs_scene_renderer import BLCSSceneRenderer

__all__ = ["BLCSSceneRenderer"]
