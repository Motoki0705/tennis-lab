"""Shared scene-style and 3D-view runtime config for task visualization.

Parses the ``visualization.style`` and ``visualization.view_3d`` Hydra
sections used identically by the BLCS and PLCS visualization CLIs into typed
runtime objects backed by the shared rendering primitives
(``src.utils.rendering``). Tasks apply only the items meaningful for them
(e.g. PLCS has no ball, so no speed/bounce HUD lines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.rendering.camera_view import CameraController
from src.utils.rendering.theme import resolve_theme

_STYLE_KEYS = frozenset(
    {"theme", "show_shadow", "show_trail", "trail_length", "show_hud", "show_minimap"}
)


@dataclass(frozen=True)
class SceneStyleConfig:
    """Resolved scene-style settings shared by task 3D scene renderers.

    Attributes:
        theme: Scene theme name (``light`` / ``dark``).
        show_shadow: Draw ground contact shadows.
        show_trail: Draw fading movement/trajectory trails.
        trail_length: Trail window length in frames.
        show_hud: Draw the HUD text overlay.
        show_minimap: Draw the top-down minimap inset.
    """

    theme: str = "light"
    show_shadow: bool = True
    show_trail: bool = True
    trail_length: int = 60
    show_hud: bool = True
    show_minimap: bool = True


def _to_plain_mapping(raw: Any, *, name: str) -> dict[str, Any]:
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"visualization.{name} must be a mapping, got {type(raw)}")
    return raw


def parse_scene_style(raw: Any) -> SceneStyleConfig:
    """Parse the ``visualization.style`` section into a typed config.

    ``None`` (section absent) yields the defaults. Unknown keys and unknown
    themes raise instead of being silently ignored.
    """
    if raw is None:
        return SceneStyleConfig()
    mapping = _to_plain_mapping(raw, name="style")

    unknown = set(mapping) - _STYLE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown visualization.style keys: {sorted(unknown)}. "
            f"Available: {sorted(_STYLE_KEYS)}"
        )

    defaults = SceneStyleConfig()
    theme = str(mapping.get("theme", defaults.theme))
    resolve_theme(theme)  # validate against the shared theme registry
    trail_length = int(mapping.get("trail_length", defaults.trail_length))
    if trail_length < 1:
        raise ValueError(f"trail_length must be >= 1, got {trail_length}")

    return SceneStyleConfig(
        theme=theme,
        show_shadow=bool(mapping.get("show_shadow", defaults.show_shadow)),
        show_trail=bool(mapping.get("show_trail", defaults.show_trail)),
        trail_length=trail_length,
        show_hud=bool(mapping.get("show_hud", defaults.show_hud)),
        show_minimap=bool(mapping.get("show_minimap", defaults.show_minimap)),
    )


def parse_view_3d(raw: Any) -> CameraController:
    """Parse the ``visualization.view_3d`` section into a camera controller.

    ``None`` (section absent) yields the static broadcast preset. See
    :meth:`src.utils.rendering.camera_view.CameraController.from_config` for
    the accepted keys.
    """
    if raw is None:
        return CameraController("broadcast")
    return CameraController.from_config(_to_plain_mapping(raw, name="view_3d"))
