"""Shared fixtures for tennis scene rendering tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tennis_scene.schema import SceneResult

SMPL_NUM_VERTICES = 6890


@pytest.fixture
def smpl_renderer_assets(tmp_path: Path) -> tuple[Path, Path]:
    """Create the two mandatory, already-resolved renderer assets."""
    faces_path = (tmp_path / "smpl_faces.npz").resolve()
    regressor_path = (tmp_path / "smpl_joint_regressor.pt").resolve()
    np.savez(faces_path, f=np.array([[0, 1, 2]], dtype=np.int64))
    torch.save(
        torch.full((1, SMPL_NUM_VERTICES), 1.0 / SMPL_NUM_VERTICES),
        regressor_path,
    )
    return faces_path, regressor_path


@pytest.fixture
def tiny_scene() -> SceneResult:
    """Minimal but schema-complete scene: 1 player, 5 frames, one ball bounce.

    SMPL vertices are all-zero (SMPL vertex count so the real joint regressor
    applies), which collapses every joint onto the player position.
    """
    num_frames = 5
    ball_z = np.array([0.5, 0.1, 0.02, 0.1, 0.5], dtype=np.float32)
    ball_3d: np.ndarray = np.zeros((num_frames, 3), dtype=np.float32)
    ball_3d[:, 0] = np.linspace(-1.0, 1.0, num_frames)
    ball_3d[:, 2] = ball_z

    return SceneResult(
        num_frames=num_frames,
        fps=30.0,
        width=1280,
        height=720,
        court_kp=np.zeros((1, num_frames, 20, 2), dtype=np.float32),
        court_vis=np.ones((1, num_frames, 20), dtype=np.float32),
        player_position=np.tile(
            np.array([2.0, -8.0, 0.0], dtype=np.float32), (1, num_frames, 1)
        ),
        player_yaw=np.zeros((1, num_frames), dtype=np.float32),
        smpl_body_pose=np.zeros((1, num_frames, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, num_frames, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        smpl_vertices_local=np.zeros(
            (1, num_frames, SMPL_NUM_VERTICES, 3), dtype=np.float32
        ),
        ball_3d=ball_3d,
        player_track_ids=np.array([0], dtype=np.int32),
    )
