"""Fixtures for structured real-dataset generation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.generate_dataset.manifest import register_exported_clip
from src.tennis_scene.schema import SceneResult
from src.utils.io import save_json_atomic


@pytest.fixture
def structured_dataset(tmp_path: Path) -> Path:
    """Create one indexed clip without requiring an actual video codec."""
    root = tmp_path / "dataset"
    clip_dir = root / "clips" / "match-001" / "clip_000"
    media_dir = clip_dir / "media"
    media_dir.mkdir(parents=True)
    (media_dir / "cam0.mp4").write_bytes(b"video-placeholder")
    manifest = {
        "version": 1,
        "clip_id": "match-001/clip_000",
        "recording_id": "match-001",
        "clip_name": "clip_000",
        "fps": 30.0,
        "num_frames": 3,
        "width": 64,
        "height": 36,
        "camera_ids": ["cam0"],
        "video_paths": ["media/cam0.mp4"],
        "cameras": [],
    }
    clip_manifest_path = save_json_atomic(manifest, clip_dir / "clip.json")
    register_exported_clip(root, clip_manifest_path)
    return root


@pytest.fixture
def valid_scene_result() -> SceneResult:
    """Return a complete BLCS+PLCS pseudo-label payload."""
    n, t, p, k = 1, 3, 2, 14
    return SceneResult(
        num_frames=t,
        fps=30.0,
        width=64,
        height=36,
        court_kp=np.zeros((n, t, k, 2), dtype=np.float32),
        court_vis=np.ones((n, t, k), dtype=np.float32),
        player_position=np.zeros((p, t, 3), dtype=np.float32),
        player_yaw=np.zeros((p, t), dtype=np.float32),
        smpl_body_pose=np.zeros((p, t, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((p, t, 3), dtype=np.float32),
        smpl_betas=np.zeros((p, 10), dtype=np.float32),
        ball_uv=np.zeros((n, t, 2), dtype=np.float32),
        ball_vis=np.ones((n, t), dtype=np.bool_),
        ball_3d=np.zeros((t, 3), dtype=np.float32),
        human_kp_2d=np.zeros((p, n, t, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((p, n, t, 17), dtype=np.float32),
        player_track_ids=np.arange(p, dtype=np.int32),
        metadata={"enabled_stages": ["plcs", "blcs"]},
    )
