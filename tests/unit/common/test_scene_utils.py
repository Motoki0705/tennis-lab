"""Unit tests for common scene utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.common.data.blcs_npz_adapter import load_camera_view
from src.common.data.camera_selection import select_camera
from src.common.data.npz_meta import decode_meta, get_num_frames
from src.common.data.scene_cache import load_npz_scene
from src.common.data.scene_paths import resolve_scene_files
from src.common.dataset.sequence import build_valid_mask, crop_to_max_len


def _write_scene(scene_dir: Path, name: str, *, num_frames: int = 5) -> Path:
    path = scene_dir / name
    np.savez(
        path,
        meta=json.dumps({"num_frames": num_frames}),
        num_cameras=np.array(1),
        cam_0_ball_uv=np.zeros((num_frames, 2), dtype=np.float32),
        cam_0_ball_visible=np.ones((num_frames,), dtype=np.float32),
        cam_0_court_kp_uv=np.zeros((20, 2), dtype=np.float32),
        cam_0_court_kp_visible=np.ones((20,), dtype=np.float32),
        ball_pos_norm=np.zeros((num_frames, 3), dtype=np.float32),
        ball_vel_world=np.zeros((num_frames, 3), dtype=np.float32),
    )
    return path


def test_scene_paths_and_meta(tmp_path: Path) -> None:
    scene_dir = tmp_path / "data"
    scenes_dir = scene_dir / "scenes"
    scenes_dir.mkdir(parents=True)
    _write_scene(scenes_dir, "scene_000.npz", num_frames=5)
    (scene_dir / "train.txt").write_text("scene_000.npz\n")

    scenes = resolve_scene_files(scene_dir, split="train", split_file=None)
    assert len(scenes) == 1
    payload = load_npz_scene(scenes[0])
    meta = decode_meta(payload["meta"])
    assert get_num_frames(meta, fallback_T=0) == 5


def test_camera_view_and_sequence_helpers(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    scene_path = _write_scene(scene_dir, "scene_001.npz", num_frames=8)

    payload = load_npz_scene(scene_path)
    view = load_camera_view(payload, 0)
    assert view.ball_uv.shape == (8, 2)

    cam_idx = select_camera("0", 1, rng=np.random.default_rng(0))
    assert cam_idx == 0

    seq_len = torch.tensor(6)
    valid_mask = build_valid_mask(8, seq_len)
    assert valid_mask.sum().item() == 6

    cropped, new_len = crop_to_max_len(
        {"ball_uv": torch.from_numpy(view.ball_uv)},
        seq_len=8,
        max_seq_len=5,
        mode="center",
    )
    assert cropped["ball_uv"].shape[0] == 5
    assert new_len == 5
