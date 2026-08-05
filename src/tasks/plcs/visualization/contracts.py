"""Lightweight input contract for PLCS scene rendering.

``PoseRenderScene`` is a duck-typing–compatible dataclass that satisfies the
attribute contract read by :class:`PLCSSceneRenderer` in comparison views
(``3d`` and ``2d_topdown``).  It can be used in place of the concrete on-disk
``Scene`` object without loading any NPZ data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PoseRenderScene:
    """Minimal scene object for PLCS renderer comparison views.

    Attributes:
        position: Normalised court position, shape ``(T, 3)``.
        rotation: Yaw encoded as ``(cos, sin)``, shape ``(T, 2)``.
        canonical_pose_3d: Joints in canonical (yaw-zero) local coordinates,
            shape ``(T, J, 3)``.  May be ``None`` when unavailable; callers
            should then restrict the renderer to ``view="2d_topdown"``.
        meta: Scene metadata dict.  Must contain ``"num_frames"`` (int).
        cameras: Camera list (may be empty for non-camera views).
        num_cameras: Number of cameras (may be 0 for non-camera views).
    """

    position: np.ndarray  # (T, 3)
    rotation: np.ndarray  # (T, 2)
    canonical_pose_3d: np.ndarray | None  # (T, J, 3) or None
    meta: dict[str, Any] = field(default_factory=dict)
    cameras: list[Any] = field(default_factory=list)
    num_cameras: int = 0
