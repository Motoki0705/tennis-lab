"""Shared scene-IO primitives for visualization orchestration.

Extracts the camera-resolution logic duplicated between the PLCS and BLCS
visualization scene loaders.  The ``num_cameras`` count and the "all cameras"
expansion differ per task and are injected by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseSceneBundle:
    """Common loaded-scene artifacts for visualization."""

    cameras: list[int]
    fps: float


def resolve_cameras(
    num_cameras: int,
    camera: int,
    cameras: list[int] | str | None,
    all_camera_fn: Callable[[], list[int]],
) -> list[int]:
    """Resolve and validate camera indices.

    Args:
        num_cameras: Number of cameras available in the scene.
        camera: Fallback single camera index.
        cameras: Optional explicit selection (list, ``"all"``, or None).
        all_camera_fn: Callable returning all camera indices for ``"all"``.

    Returns:
        Validated list of selected camera indices.
    """
    if cameras == "all":
        selected = all_camera_fn()
    elif cameras:
        selected = list(cameras)
    else:
        selected = [camera]

    if not selected:
        raise ValueError("No cameras selected.")

    for cam_idx in selected:
        if cam_idx < 0 or cam_idx >= num_cameras:
            raise ValueError(
                f"Camera {cam_idx} out of range (0-{num_cameras - 1})."
            )
    return selected
