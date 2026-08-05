"""Strict shared scene-style and 3D-view runtime contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import (
    ConfigMapping,
    exact_config_mapping,
    require_config_value,
)
from src.utils.configuration import (
    SemanticConfigurationError,
)
from src.utils.rendering.camera_view import (
    CameraController,
    CameraKeyframe,
    CameraMode,
    CameraView3D,
    resolve_camera_view,
)
from src.utils.rendering.theme import resolve_theme

_STYLE_KEYS = frozenset(
    {"theme", "show_shadow", "show_trail", "trail_length", "show_hud", "show_minimap"}
)
_VIEW_REQUIRED_KEYS = frozenset({"mode", "orbit_period_s", "keyframes"})
_VIEW_POSE_KEYS = frozenset({"preset", "elev", "azim", "zoom"})
_VIEW_KEYS = _VIEW_REQUIRED_KEYS | _VIEW_POSE_KEYS
_CAMERA_MODES = frozenset({"static", "orbit", "keyframes"})


@dataclass(frozen=True, slots=True)
class SceneStyleConfig:
    """Exact scene-style values selected by composed configuration."""

    theme: str
    show_shadow: bool
    show_trail: bool
    trail_length: int
    show_hud: bool
    show_minimap: bool


def _to_plain_mapping(raw: object, *, name: str) -> ConfigMapping:
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    return exact_config_mapping(
        raw,
        path=f"visualization.{name}",
        required_keys=(_STYLE_KEYS if name == "style" else _VIEW_REQUIRED_KEYS),
        optional_keys=(frozenset() if name == "style" else _VIEW_POSE_KEYS),
    )


def _number_or_none(mapping: ConfigMapping, key: str, *, path: str) -> float | None:
    if key not in mapping:
        return None
    value = require_config_value(mapping, key, (float, int, type(None)), path=path)
    if value is None:
        return None
    return float(cast("float | int", value))


def _parse_view_pose(mapping: ConfigMapping, *, path: str) -> CameraView3D:
    preset = cast(
        "str | None",
        (
            require_config_value(mapping, "preset", (str, type(None)), path=path)
            if "preset" in mapping
            else None
        ),
    )
    elev = _number_or_none(mapping, "elev", path=path)
    azim = _number_or_none(mapping, "azim", path=path)
    zoom = _number_or_none(mapping, "zoom", path=path)

    if preset is not None and (elev is not None or azim is not None):
        raise SemanticConfigurationError(
            f"{path}: specify either preset or explicit elev/azim, not both."
        )
    if preset is not None:
        base = resolve_camera_view(preset)
    elif elev is not None and azim is not None:
        if zoom is None:
            raise SemanticConfigurationError(
                f"{path}: explicit elev/azim requires an explicit zoom."
            )
        base = CameraView3D(elev=elev, azim=azim, zoom=zoom)
    else:
        raise SemanticConfigurationError(
            f"{path}: requires preset or both elev and azim."
        )
    return CameraView3D(
        elev=base.elev,
        azim=base.azim,
        zoom=base.zoom if zoom is None else zoom,
    )


def parse_scene_style(raw: object) -> SceneStyleConfig:
    """Return the exact-typed ``visualization.style`` contract."""
    mapping = _to_plain_mapping(raw, name="style")
    theme = cast(
        "str", require_config_value(mapping, "theme", str, path="visualization.style")
    )
    resolve_theme(theme)
    trail_length = cast(
        "int",
        require_config_value(mapping, "trail_length", int, path="visualization.style"),
    )
    if trail_length < 1:
        raise SemanticConfigurationError(
            f"visualization.style.trail_length must be >= 1; got {trail_length}."
        )
    return SceneStyleConfig(
        theme=theme,
        show_shadow=cast(
            "bool",
            require_config_value(
                mapping, "show_shadow", bool, path="visualization.style"
            ),
        ),
        show_trail=cast(
            "bool",
            require_config_value(
                mapping, "show_trail", bool, path="visualization.style"
            ),
        ),
        trail_length=trail_length,
        show_hud=cast(
            "bool",
            require_config_value(mapping, "show_hud", bool, path="visualization.style"),
        ),
        show_minimap=cast(
            "bool",
            require_config_value(
                mapping, "show_minimap", bool, path="visualization.style"
            ),
        ),
    )


def parse_view_3d(raw: object) -> CameraController:
    """Return an exact recursive ``visualization.view_3d`` contract."""
    mapping = _to_plain_mapping(raw, name="view_3d")
    mode = cast(
        "str", require_config_value(mapping, "mode", str, path="visualization.view_3d")
    )
    if mode not in _CAMERA_MODES:
        raise SemanticConfigurationError(
            f"visualization.view_3d.mode must be one of {sorted(_CAMERA_MODES)}; "
            f"got {mode!r}."
        )
    orbit_period = float(
        cast(
            "float | int",
            require_config_value(
                mapping,
                "orbit_period_s",
                (float, int),
                path="visualization.view_3d",
            ),
        )
    )
    raw_keyframes = require_config_value(
        mapping,
        "keyframes",
        (list, tuple, type(None)),
        path="visualization.view_3d",
    )
    keyframes: list[CameraKeyframe] | None = None
    if raw_keyframes is not None:
        keyframes = []
        for index, raw_keyframe in enumerate(cast("Sequence[object]", raw_keyframes)):
            path = f"visualization.view_3d.keyframes[{index}]"
            keyframe = exact_config_mapping(
                raw_keyframe,
                path=path,
                required_keys=frozenset({"frame"}),
                optional_keys=_VIEW_POSE_KEYS,
            )
            keyframes.append(
                CameraKeyframe(
                    frame=cast(
                        "int",
                        require_config_value(keyframe, "frame", int, path=path),
                    ),
                    view=_parse_view_pose(keyframe, path=path),
                )
            )
    return CameraController(
        _parse_view_pose(mapping, path="visualization.view_3d"),
        mode=cast("CameraMode", mode),
        orbit_period_s=orbit_period,
        keyframes=keyframes,
    )


__all__ = ["SceneStyleConfig", "parse_scene_style", "parse_view_3d"]
