"""Shared visualization orchestration primitives.

Extracts the runtime-config scaffolding duplicated between the PLCS and BLCS
visualization orchestrators: the resolved runtime config dataclass, ``auto``
device resolution, Hydra camera-selection parsing, and animation save/show.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt

from src.tasks.base.configuration import (
    SceneVisualizationConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.tasks.base.visualization.style import (
    SceneStyleConfig,
    parse_scene_style,
    parse_view_3d,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.device import resolve_device as _resolve_torch_device
from src.utils.paths import PROJECT_ROOT
from src.utils.rendering.camera_view import CameraController

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseVisualizationRuntimeConfig:
    """Resolved runtime settings shared by task visualization orchestrators.

    ``style`` and ``view_3d`` drive the shared rich 3D rendering; ``camera``
    and ``cameras`` keep their existing meaning as *input* scene-camera
    selectors and are unrelated to the 3D viewpoint.
    """

    mode: str
    scene_path: Path
    checkpoint: Path | None
    device: str
    animation_view: str
    fps: float | None
    save: Path | None
    camera: int
    cameras: tuple[int, ...] | Literal["all"] | None
    info: bool
    style: SceneStyleConfig
    view_3d: CameraController


def resolve_device(device: str) -> str:
    """Resolve ``auto`` device selection.

    Delegates the ``auto`` resolution to :func:`src.utils.device.resolve_device`;
    explicit device strings are passed through unchanged (no CPU fallback).
    """
    if device == "auto":
        return str(_resolve_torch_device("auto"))
    return device


def parse_hw(value: object, *, name: str) -> tuple[int, int]:
    """Parse a length-2 ``(height, width)`` int pair from a Hydra config value."""
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be exactly list or tuple.")
    sequence = cast("Sequence[object]", value)
    if len(sequence) != 2 or any(type(item) is not int for item in sequence):
        raise TypeError(f"{name} must contain exactly two int values.")
    return cast("int", sequence[0]), cast("int", sequence[1])


def parse_rgb(value: object, *, name: str) -> tuple[int, int, int]:
    """Parse a length-3 RGB int triple from a Hydra config value."""
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be exactly list or tuple.")
    sequence = cast("Sequence[object]", value)
    if len(sequence) != 3 or any(type(item) is not int for item in sequence):
        raise TypeError(f"{name} must contain exactly three int values.")
    rgb = cast("tuple[int, int, int]", tuple(sequence))
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"{name} channels must be within [0, 255].")
    return rgb


def parse_float_triplet(value: object, *, name: str) -> tuple[float, float, float]:
    """Parse a length-3 float triple from a Hydra config value."""
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be exactly list or tuple.")
    sequence = cast("Sequence[object]", value)
    if len(sequence) != 3 or any(type(item) not in (float, int) for item in sequence):
        raise TypeError(f"{name} must contain exactly three numeric values.")
    return (
        float(cast("float | int", sequence[0])),
        float(cast("float | int", sequence[1])),
        float(cast("float | int", sequence[2])),
    )


def save_or_show_animation(
    anim: Any,
    save: Path | None,
    fps: float,
) -> None:
    """Save the animation to ``save`` or show it interactively."""
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save), fps=fps)
        plt.close()
        logger.info(f"Saved animation to {save}")
    else:
        plt.show()


def build_scene_runtime_config(
    cfg: object,
    *,
    visualization_extension_keys: Set[str] = frozenset(),
) -> BaseVisualizationRuntimeConfig:
    """Build the shared scene-visualization runtime config from a Hydra config.

    Common to the PLCS and BLCS orchestrators: resolves scene/checkpoint/save
    paths to absolute, resolves the device (preferring ``run.device`` then
    ``visualization.device``), and parses the Hydra camera selection.
    """
    root = as_config_mapping(cfg, path="configuration")
    vis = require_config_mapping(root, "visualization", path="configuration")
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            require_config_mapping(root, "paths", path="configuration"),
            repository_root=PROJECT_ROOT,
        )
    )
    common = SceneVisualizationConfig.from_mapping(
        vis,
        resolver=resolver,
        extension_keys=visualization_extension_keys,
    )

    return BaseVisualizationRuntimeConfig(
        mode=common.mode,
        scene_path=common.scene_path,
        checkpoint=common.checkpoint,
        device=resolve_device(common.device),
        animation_view=common.animation_view,
        fps=common.fps,
        save=common.save,
        camera=common.camera,
        cameras=common.cameras,
        info=common.info,
        style=parse_scene_style(
            require_config_mapping(vis, "style", path="visualization")
        ),
        view_3d=parse_view_3d(
            require_config_mapping(vis, "view_3d", path="visualization")
        ),
    )
