"""Synthetic TrackNet, Meiji and chat-annotation ball sources (test support)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from omegaconf import OmegaConf

from src.tasks.ball_detection.generate_dataset.frame_store.config import (
    FrameStoreBuildConfig,
)
from src.tennis_scene.chat_annotation.runtime.contracts import (
    Ball,
    BallAnnotation,
    ClipManifest,
    make_template,
    sha256_file,
    write_json,
)
from src.tennis_scene.chat_annotation.runtime.media import encode_video
from tests.support.tasks.player_detection.chat_root import write_clip

TRACKNET_SIZE = (64, 48)  # width, height
MEIJI_SIZE = (96, 64)
MEIJI_FRAMES = 5
MEIJI_TIME_BASE = Fraction(1, 2997003)
MEIJI_PTS_STEP = 50000

# (visibility, x, y, status) per TrackNet frame.
TRACKNET_ROWS = [("1", 10, 12, "0"), ("2", 20, 14, "1"), ("0", 0, 0, ""), ("3", 30, 16, "2")]


def tracknet_pixels(index: int) -> np.ndarray:
    width, height = TRACKNET_SIZE
    image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 30 * index
    image[5:15, 5 + 4 * index : 15 + 4 * index] = 240
    return image


def write_tracknet(root: Path, games: tuple[str, ...] = ("game1", "game2", "game3")) -> None:
    for game in games:
        clip = root / game / "Clip1"
        clip.mkdir(parents=True)
        lines = ["file name,visibility,x-coordinate,y-coordinate,status"]
        for index, (visibility, x, y, status) in enumerate(TRACKNET_ROWS):
            name = f"{index:04d}.jpg"
            cv2.imwrite(str(clip / name), tracknet_pixels(index))
            lines.append(f"{name},{visibility},{x},{y},{status}")
        (clip / "Label.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def meiji_pixels(index: int) -> np.ndarray:
    width, height = MEIJI_SIZE
    image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 200, width, dtype=np.uint8)[None, :]
    image[20:40, 10 + 8 * index : 30 + 8 * index] = 250
    return image


def meiji_row(index: int, status: str, xy: tuple[float, float] | None, *, break_before: bool = False) -> dict[str, Any]:
    width, height = MEIJI_SIZE
    visibility = {
        "observed": "visible",
        "interpolated": "not_independently_visible",
        "occlusion_estimated": "occluded",
        "unresolved": "unknown",
    }[status]
    return {
        "frame_index": index,
        "pts": index * MEIJI_PTS_STEP,
        "track_id": 1,
        "label": "tennis_ball",
        "status": status,
        "visibility": visibility,
        "center_px": None if xy is None else {"x": xy[0], "y": xy[1]},
        "center_normalized": None if xy is None else {"x": xy[0] / width, "y": xy[1] / height},
        "break_before": break_before,
    }


MEIJI_ROWS = [
    ("observed", (12.0, 30.0), False),
    ("interpolated", (20.0, 30.0), False),
    ("occlusion_estimated", (28.0, 31.0), True),
    ("unresolved", None, False),
    ("observed", (44.0, 33.0), False),
]


def write_meiji(root: Path, videos: tuple[str, ...] = ("video_000", "video_001")) -> None:
    """``dataset.json`` with one two-camera clip per video, each camera annotated."""
    width, height = MEIJI_SIZE
    entries = []
    for video_id in videos:
        clip_dir = root / "videos" / video_id / "clips" / "clip_000"
        (clip_dir / "media").mkdir(parents=True)
        (clip_dir / "outsource").mkdir()
        cameras = []
        for camera_id in ("cam0", "cam1"):
            media = clip_dir / "media" / f"{camera_id}.mp4"
            encode_video(
                media,
                width=width,
                height=height,
                time_base=MEIJI_TIME_BASE,
                rate=1 / (MEIJI_TIME_BASE * MEIJI_PTS_STEP),
                frames=[
                    (av.VideoFrame.from_ndarray(meiji_pixels(i), format="rgb24"), i * MEIJI_PTS_STEP, MEIJI_PTS_STEP)
                    for i in range(MEIJI_FRAMES)
                ],
                crf=0,
                preset="ultrafast",
            )
            rows = [meiji_row(i, status, xy, break_before=brk) for i, (status, xy, brk) in enumerate(MEIJI_ROWS)]
            counts = {kind: sum(1 for row in rows if row["status"] == kind) for kind in
                      ("observed", "interpolated", "occlusion_estimated", "unresolved")}
            annotation = {
                "schema_version": "video_ball_annotation.v2",
                "source": {
                    "file_name": media.name,
                    "sha256": sha256_file(media),
                    "width": width,
                    "height": height,
                    "fps_numerator": 2997003,
                    "fps_denominator": MEIJI_PTS_STEP,
                    "frame_count": MEIJI_FRAMES,
                    "time_base": str(MEIJI_TIME_BASE),
                    "rotation": 0,
                },
                "coordinate_system": {
                    "origin": "top_left",
                    "frame_index": "zero_based",
                    "normalization": {"x": "x_px / width", "y": "y_px / height"},
                },
                "target": {"track_id": 1, "label": "tennis_ball"},
                "review": {"reviewer": "synthetic"},
                "summary": {"counts": counts},
                "frames": rows,
            }
            (clip_dir / "outsource" / f"{camera_id}_annotations.json").write_text(json.dumps(annotation))
            cameras.append({"camera_id": camera_id, "video": f"media/{camera_id}.mp4"})
        clip_id = f"{video_id}/clip_000"
        (clip_dir / "clip.json").write_text(json.dumps({"clip_id": clip_id, "video_id": video_id, "cameras": cameras}))
        entries.append(
            {"clip_id": clip_id, "video_id": video_id, "clip_name": "clip_000", "path": f"videos/{video_id}/clips/clip_000"}
        )
    (root / "dataset.json").write_text(json.dumps({"clips": entries}))


def publish_ball_annotation(root: Path, manifest: ClipManifest, balls: dict[int, list[Ball]], *,
                            unreviewed: tuple[int, ...] = (), breaks: tuple[int, ...] = ()) -> None:
    annotation = make_template(manifest, "ball")
    assert isinstance(annotation, BallAnnotation)
    annotation.issues = ["synthetic fixture"]
    for frame in annotation.frames:
        frame.reviewed = frame.frame_index not in unreviewed
        frame.balls = balls.get(frame.frame_index, [])
        frame.interpolation_break = frame.frame_index in breaks
    write_json(root / "annotated" / "processed" / "ball" / f"{annotation.clip_id}.json", annotation.model_dump())


def ball(status: str, xy: list[float] | None, track: str = "ball_001") -> Ball:
    return Ball.model_validate({"track_id": track, "center_px": xy, "status": status, "interpolation_frames": None})


def write_chat(root: Path) -> ClipManifest:
    """One 96x64 six-frame clip with every ball status except ``interpolated``."""
    manifest = write_clip(root, "src_a", "f000-006")
    publish_ball_annotation(
        root,
        manifest,
        {
            0: [ball("visible", [10.0, 20.0])],
            1: [ball("occluded", [14.0, 21.0])],
            2: [ball("occluded", None)],
            3: [ball("out_of_frame", None)],
            4: [ball("unresolved", None), ball("visible", [50.0, 30.0], track="ball_002")],
        },
        unreviewed=(5,),
        breaks=(1,),
    )
    return manifest


@dataclass(frozen=True)
class SourceRoots:
    tmp_path: Path

    @property
    def data(self) -> Path:
        return self.tmp_path / "data"

    @property
    def outputs(self) -> Path:
        return self.tmp_path / "outputs"

    def config(self, sources: dict[str, Any], *, max_height: int = 720, version: str = "test-v1") -> FrameStoreBuildConfig:
        raw = {
            "paths": {
                "project_root": str(self.tmp_path),
                "data_root": str(self.data),
                "checkpoint_root": str(self.tmp_path / "ckpt"),
                "artifact_root": str(self.outputs),
                "output_root": str(self.outputs),
                "cache_root": str(self.tmp_path / "cache"),
                "external_asset_root": str(self.tmp_path / "third_party"),
            },
            "sources": sources,
            "dataset": {"version": version, "jpeg_quality": 95, "max_height": max_height},
            "workers": 1,
            "run": {"output_dir": f"ball_detection/{version}"},
        }
        return FrameStoreBuildConfig.from_config(OmegaConf.create(raw))


TRACKNET_SOURCE = {
    "root": "tennis/tracknet",
    "fps": 30,
    "split": {"train": ["game1"], "val": ["game2"], "test": ["game3"]},
}
MEIJI_SOURCE = {"root": "meiji", "split": {"train": ["video_000"], "val": [], "test": ["video_001"]}}
CHAT_SOURCE = {
    "annotation_root": "chat_annotation",
    "allowed_statuses": ["completed", "partial"],
    "split": {"val_ratio": 0.15, "test_ratio": 0.15, "seed": 0},
}
