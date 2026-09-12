"""Video/annotation alignment, missing-label semantics and clip-grouped splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.ball_detection.configuration import validate_training

CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/ball_detection/configs"


def make_multiview_config(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(
            config_name="train_meiji_3cam", overrides=["data/augmentation=none"]
        )
    cfg.paths.data_root = str(tmp_path / "data")
    cfg.paths.cache_root = str(tmp_path / "cache")
    cfg.paths.output_root = str(tmp_path / "output")
    cfg.data.data_dir = "fixture"
    cfg.data.image_size = [32, 64]
    cfg.data.heatmap_size = [32, 64]
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.sample_stride = 1
    cfg.model.num_frames = 2
    cfg.data.split.root_role = "data"
    root = tmp_path / "data" / "fixture"
    clips = []
    for split, recording in zip(
        ("train", "val", "test"), ("r0", "r1", "r2"), strict=True
    ):
        clip_id = f"{recording}/clip_000"
        entry = {
            "clip_id": clip_id,
            "path": f"clips/{clip_id}",
            "num_frames": 10,
            "width": 80,
            "height": 48,
            "num_cameras": 2,
        }
        clips.append(entry)
        clip_dir = root / str(entry["path"])
        (clip_dir / "media").mkdir(parents=True)
        (clip_dir / "outsource").mkdir()
        cameras = []
        for camera in ("cam0", "cam1"):
            video_path = clip_dir / "media" / f"{camera}.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (80, 48)
            )
            assert writer.isOpened()
            for index in range(10):
                writer.write(np.full((48, 80, 3), 20 + 15 * index, dtype=np.uint8))
            writer.release()
            frames = [
                {
                    "frame_index": i,
                    "track_id": 1,
                    "label": "tennis_ball",
                    "status": status,
                    "center_px": None
                    if status == "unresolved"
                    else {"x": 40.0, "y": 24.0},
                    "break_before": i in {0, 7},
                }
                for i, status in enumerate(
                    (
                        "observed",
                        "interpolated",
                        "occlusion_estimated",
                        "observed",
                        "unresolved",
                        "observed",
                        "observed",
                        "observed",
                        "observed",
                        "observed",
                    )
                )
            ]
            annotation = {
                "schema_version": "video_ball_annotation.v2",
                "target": {"track_id": 1},
                "source": {
                    "file_name": video_path.name,
                    "width": 80,
                    "height": 48,
                    "frame_count": 10,
                    "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                },
                "coordinate_system": {
                    "origin": "top_left",
                    "x_axis": "right",
                    "y_axis": "down",
                    "frame_index": "zero_based",
                },
                "frames": frames,
            }
            (clip_dir / "outsource" / f"{camera}_annotations.json").write_text(
                json.dumps(annotation)
            )
            cameras.append({"camera_id": camera, "video": f"media/{camera}.mp4"})
        (clip_dir / "clip.json").write_text(
            json.dumps({**entry, "camera_ids": ["cam0", "cam1"], "cameras": cameras})
        )
        split_path = root / f"{split}.txt"
        split_path.write_text(clip_id + "\n")
        cfg.data.split[f"{split}_file"] = f"fixture/{split}.txt"
    (root / "dataset.json").write_text(json.dumps({"version": 1, "clips": clips}))
    validate_training(cfg)
    return cfg
