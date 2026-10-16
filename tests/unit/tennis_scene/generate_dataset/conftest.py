"""Fixtures for structured real-dataset generation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.generate_dataset.manifest import register_exported_clip
from src.tennis_scene.schema import SceneResult
from src.utils.io import save_json_atomic


@pytest.fixture
def structured_dataset(tmp_path: Path) -> Path:
    """Create one indexed clip without requiring an actual video codec."""
    root = tmp_path / "dataset"
    clip_dir = root / "videos" / "video_000" / "clips" / "clip_000"
    media_dir = clip_dir / "media"
    media_dir.mkdir(parents=True)
    (media_dir / "cam0.mp4").write_bytes(b"video-placeholder")
    manifest = {
        "version": 2,
        "dataset_id": "test-dataset",
        "clip_id": "video_000/clip_000",
        "video_id": "video_000",
        "clip_name": "clip_000",
        "fps": 30.0,
        "num_frames": 3,
        "width": 64,
        "height": 36,
        "global_start_sec": 0.0,
        "global_end_sec": 0.1,
        "camera_ids": ["cam0"],
        "video_paths": ["media/cam0.mp4"],
        "cameras": [],
        "sync_source": "clip_studio",
        "exported_at": "2026-09-11T00:00:00+00:00",
    }
    clip_manifest_path = save_json_atomic(manifest, clip_dir / "clip.json")
    register_exported_clip(root, clip_manifest_path)
    return root


@pytest.fixture
def valid_scene_result() -> SceneResult:
    """Return a complete declared-pipeline v2 scene for the one-camera clip."""
    n, t, p, k = 1, 3, 2, 14
    valid: NDArray[np.bool_] = np.ones((p, t), bool)
    return SceneResult(
        num_frames=t,
        fps=30.0,
        width=64,
        height=36,
        court_kp=np.zeros((n, t, k, 2), dtype=np.float32),
        court_vis=np.ones((n, t, k), dtype=np.float32),
        player_position=np.zeros((p, t, 3), dtype=np.float32),
        player_yaw=np.zeros((p, t), dtype=np.float32),
        ball_uv=np.zeros((n, t, 2), dtype=np.float32),
        ball_vis=np.ones((n, t), dtype=np.bool_),
        ball_3d=np.zeros((t, 3), dtype=np.float32),
        human_kp_2d=np.zeros((p, n, t, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((p, n, t, 17), dtype=np.float32),
        player_track_ids=np.arange(p, dtype=np.int32),
        player_kp_3d=np.zeros((p, t, 17, 3), dtype=np.float32),
        player_observed=valid.copy(),
        player_valid=valid.copy(),
        player_heading_valid=valid.copy(),
        player_kp_3d_vis=np.ones((p, t, 17), bool),
        player_smpl_valid=np.zeros((p, t), bool),
        ball_3d_valid=np.zeros(t, bool),  # one camera cannot triangulate
        player_rejection_code=np.zeros((p, t), np.uint8),
        player_kp_3d_rejection_code=np.zeros((p, t, 17), np.uint8),
        ball_rejection_code=np.ones(t, np.uint8),
        metadata={"scene_schema_version": 2, "pipeline_contract": "declared_components_v1", "status": "ok",
                  "validity_statistics": {}, "dataset_clip_id": "video_000/clip_000",
                  "court_reference": {"camera_ids": ["cam0"]}},
    )
