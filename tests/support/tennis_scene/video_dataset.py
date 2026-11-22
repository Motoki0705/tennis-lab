"""Small structured clips shared by video-selection and batch-entry tests."""

from __future__ import annotations

from pathlib import Path

from src.tennis_scene.generate_dataset.manifest import register_exported_clip
from src.utils.io import save_json_atomic


def add_clip(dataset: Path, clip_id: str, *, frames: int = 24, cameras: int = 3) -> Path:
    video, name = clip_id.split("/")
    clip = dataset / "videos" / video / "clips" / name
    clip.mkdir(parents=True)
    camera_ids = [f"cam{i}" for i in range(cameras)]
    for camera in camera_ids:
        (clip / f"{camera}.mp4").write_bytes(f"{clip_id}/{camera}".encode())
    path = save_json_atomic({
        "version": 2, "dataset_id": "batch-fixture", "clip_id": clip_id, "video_id": video, "clip_name": name,
        "fps": 30., "num_frames": frames, "width": 1280, "height": 720,
        "global_start_sec": 0., "global_end_sec": frames / 30,
        "camera_ids": camera_ids, "video_paths": [f"{camera}.mp4" for camera in camera_ids],
        "cameras": [], "sync_source": "clip_studio", "exported_at": "2026-11-17T00:00:00Z",
    }, clip / "clip.json")
    register_exported_clip(dataset, path)
    return clip
