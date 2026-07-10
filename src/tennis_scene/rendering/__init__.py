"""Tennis scene rendering module.

Camera, theme, layer, HUD, and minimap primitives live in
``src.utils.rendering``; this package holds only the ``SceneResult``-aware
:class:`TennisSceneRenderer`.
"""

from src.tennis_scene.rendering.tennis_scene_renderer import (
    TennisSceneRenderer,
    TennisSceneStyle,
)

__all__ = [
    "TennisSceneRenderer",
    "TennisSceneStyle",
]
