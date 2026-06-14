"""Shared visualization orchestration primitives.

Extracts the runtime-config scaffolding duplicated between the PLCS and BLCS
visualization orchestrators: the resolved runtime config dataclass, ``auto``
device resolution, Hydra camera-selection parsing, and animation save/show.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseVisualizationRuntimeConfig:
    """Resolved runtime settings shared by task visualization orchestrators."""

    mode: str
    scene_path: Path
    checkpoint: Path | None
    device: str
    animation_view: str
    fps: float | None
    save: Path | None
    camera: int
    cameras: list[int] | str | None
    info: bool


def resolve_device(device: str) -> str:
    """Resolve ``auto`` device selection."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_cameras(raw_value: object) -> list[int] | str | None:
    """Parse Hydra camera selection value into optional list[int]."""
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return None
        if stripped == "all":
            return "all"
        return [int(part.strip()) for part in stripped.split(",")]
    return [int(v) for v in cast("Iterable[Any]", raw_value)]


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
