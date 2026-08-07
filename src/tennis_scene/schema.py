"""Canonical schema for an integrated tennis-scene reconstruction result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SceneResult:
    """Result of tennis scene 3D reconstruction.

    Player-related arrays use ``(P, T, ...)`` as the canonical shape. Camera
    observations use a leading camera axis ``N``.

    ``player_position`` and ``ball_3d`` are in court coordinates: XY is the
    court plane and +Z is up. ``smpl_vertices_local`` and SMPL pose parameters
    are stored in the GVHMR/SMPL body convention; rendering root-centers the
    vertices and explicitly converts Y-up SMPL geometry to court Z-up before
    applying ``player_yaw``.

    Archive persistence is intentionally separate from this schema. Use
    :func:`src.tennis_scene.archive.save_scene_result` and
    :func:`src.tennis_scene.archive.load_scene_result`.
    """

    num_frames: int
    fps: float
    width: int
    height: int

    court_kp: NDArray[np.float32]  # (N, T, K, 2)
    court_vis: NDArray[np.float32]  # (N, T, K)

    player_position: NDArray[np.float32]  # (P, T, 3)
    player_yaw: NDArray[np.float32]  # (P, T)

    smpl_body_pose: NDArray[np.float32]  # (P, T, 63)
    smpl_global_orient: NDArray[np.float32]  # (P, T, 3)
    smpl_betas: NDArray[np.float32]  # (P, 10)
    smpl_vertices_local: NDArray[np.float32] | None = None  # (P, T, V, 3)

    ball_uv: NDArray[np.float32] | None = None  # (N, T, 2)
    ball_vis: NDArray[np.bool_] | None = None  # (N, T)
    ball_3d: NDArray[np.float32] | None = None  # (T, 3)

    human_kp_2d: NDArray[np.float32] | None = None  # (P, N, T, 17, 2)
    human_kp_vis: NDArray[np.float32] | None = None  # (P, N, T, 17)

    player_track_ids: NDArray[np.int32] | None = None
    player_kp_3d: NDArray[np.float32] | None = None  # (P, T, J, 3)

    metadata: dict[str, Any] = field(default_factory=dict)
