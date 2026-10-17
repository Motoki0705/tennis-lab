"""Staged scene reveal: capture cameras -> SfM points -> NHT Gaussians -> render."""

from src.synthetic_data_generation.visualization.reveal.contracts import (
    SCENE_REVEAL_METADATA_SCHEMA,
    RevealMode,
    RevealPhase,
    SceneRevealRequest,
    SceneRevealResult,
)
from src.synthetic_data_generation.visualization.reveal.renderer import (
    render_scene_reveal,
)

__all__ = [
    "SCENE_REVEAL_METADATA_SCHEMA",
    "RevealMode",
    "RevealPhase",
    "SceneRevealRequest",
    "SceneRevealResult",
    "render_scene_reveal",
]
