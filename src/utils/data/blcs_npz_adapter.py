"""Adapters for BLCS-style NPZ scene payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraView:
    """Container for per-camera NPZ arrays."""

    ball_uv: np.ndarray
    ball_vis: np.ndarray
    court_kp: np.ndarray
    court_vis: np.ndarray


def _require_keys(data: Mapping[str, Any], keys: Sequence[str], *, context: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        available = ", ".join(sorted(data.keys()))
        missing_str = ", ".join(missing)
        raise KeyError(
            f"Missing {context} keys: {missing_str}. Available keys: {available}"
        )


def load_camera_view(data: Mapping[str, Any], cam_idx: int) -> CameraView:
    """Load per-camera arrays from a scene payload."""
    prefix = f"cam_{cam_idx}_"
    keys = (
        f"{prefix}ball_uv",
        f"{prefix}ball_visible",
        f"{prefix}court_kp_uv",
        f"{prefix}court_kp_visible",
    )
    _require_keys(data, keys, context=f"camera {cam_idx}")
    return CameraView(
        ball_uv=np.asarray(data[keys[0]]),
        ball_vis=np.asarray(data[keys[1]]),
        court_kp=np.asarray(data[keys[2]]),
        court_vis=np.asarray(data[keys[3]]),
    )


def require_3d_keys(data: Mapping[str, Any], keys: Sequence[str]) -> None:
    """Ensure required 3D keys exist in the payload."""
    _require_keys(data, keys, context="3D trajectory")


def load_3d_arrays(
    data: Mapping[str, Any],
    *,
    position_norm_key: str | None = "ball_pos_norm",
    position_world_key: str | None = None,
    velocity_world_key: str | None = None,
) -> dict[str, np.ndarray]:
    """Load 3D arrays with clear error messages when missing."""
    keys: list[str] = []
    if position_norm_key:
        keys.append(position_norm_key)
    if position_world_key:
        keys.append(position_world_key)
    if velocity_world_key:
        keys.append(velocity_world_key)
    require_3d_keys(data, keys)
    payload: dict[str, np.ndarray] = {}
    if position_norm_key:
        payload[position_norm_key] = np.asarray(data[position_norm_key])
    if position_world_key:
        payload[position_world_key] = np.asarray(data[position_world_key])
    if velocity_world_key:
        payload[velocity_world_key] = np.asarray(data[velocity_world_key])
    return payload
