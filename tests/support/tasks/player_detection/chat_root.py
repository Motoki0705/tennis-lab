"""Synthetic prepared chat-annotation root with real encoded clips (test support)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np
from omegaconf import OmegaConf

from src.tasks.player_detection.configuration import GenerateDatasetConfig
from src.tennis_scene.chat_annotation.layout import video_path
from src.tennis_scene.chat_annotation.runtime.contracts import (
    KIT_VERSION,
    ClipManifest,
    FrameMap,
    FrameRange,
    Player,
    PlayerAnnotation,
    Policies,
    SourceInfo,
    make_template,
    sha256_file,
    write_json,
)
from src.tennis_scene.chat_annotation.runtime.media import encode_video, probe_video

WIDTH, HEIGHT, FRAMES = 96, 64, 6


def frame_pixels(index: int) -> np.ndarray:
    """Distinct RGB frames: gradient background plus a bright moving block."""
    image: np.ndarray = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 200, WIDTH, dtype=np.uint8)[None, :]
    image[..., 1] = 40 + 20 * index
    image[10:40, 10 + 5 * index : 30 + 5 * index] = 250
    return image


def write_clip(root: Path, source_id: str, clip_id: str) -> ClipManifest:
    """Encode a real lossless clip and derive its manifest from the probe."""
    filename = f"{source_id}__run__{clip_id}.mp4"
    video = root / "videos" / source_id / filename
    video.parent.mkdir(parents=True)
    encode_video(
        video,
        width=WIDTH,
        height=HEIGHT,
        time_base=Fraction(1, 30),
        rate=Fraction(30),
        frames=[(av.VideoFrame.from_ndarray(frame_pixels(i), format="rgb24"), i, 1) for i in range(FRAMES)],
        crf=0,
        preset="ultrafast",
    )
    timeline = probe_video(video)
    assert len(timeline.pts) == FRAMES
    manifest = ClipManifest(
        schema_version="tennis_chat_clip.v1",
        kit_version=KIT_VERSION,
        kit_id="c" * 64,
        clip_id=clip_id,
        source=SourceInfo(
            source_id=source_id,
            youtube_id=None,
            url=None,
            title=f"Synthetic {source_id}",
            filename=f"{source_id}.mp4",
            sha256="a" * 64,
            bytes=1000,
            acquired_at="2026-09-26",
        ),
        filename=filename,
        sha256=sha256_file(video),
        bytes=video.stat().st_size,
        width=WIDTH,
        height=HEIGHT,
        time_base=str(timeline.time_base),
        nominal_fps=str(timeline.rate),
        source_start_pts=timeline.pts[0],
        media_range=FrameRange(start=0, stop=FRAMES),
        target_range=FrameRange(start=0, stop=FRAMES),
        frames=[
            FrameMap(
                frame_index=i,
                source_frame_index=i,
                source_pts=pts,
                clip_pts=pts - timeline.pts[0],
                duration_pts=duration,
                is_target=True,
            )
            for i, (pts, duration) in enumerate(zip(timeline.pts, timeline.durations, strict=True))
        ],
        policies=Policies(ball_max_gap_seconds=0.1),
    )
    assert video == video_path(root, manifest)
    directory = root / "_preparation" / source_id / "run" / "clips" / clip_id
    write_json(directory / "clip_manifest.json", manifest.model_dump())
    write_json(
        directory.parents[1] / "ready" / f"{clip_id}.json",
        {"files": {"clip_manifest.json": sha256_file(directory / "clip_manifest.json"), manifest.filename: manifest.sha256}},
    )
    return manifest


def player(track: str, box: list[float] | None, *, truncated: bool = False) -> Player:
    source: Literal["observed", "inferred", "unresolved"] = (
        "unresolved" if box is None else ("inferred" if truncated else "observed")
    )
    return Player(track_id=track, bbox_xyxy=box, bbox_source=source, occluded=False, truncated=truncated)


def publish_player_annotation(root: Path, manifest: ClipManifest, players: dict[int, list[Player]]) -> None:
    annotation = make_template(manifest, "player")
    assert isinstance(annotation, PlayerAnnotation)
    annotation.issues = ["synthetic fixture: spatial precision not verified"]
    for frame in annotation.frames:
        frame.reviewed = True
        frame.players = players.get(frame.frame_index, [])
    path = root / "annotated" / "processed" / "player" / f"{annotation.clip_id}.json"
    write_json(path, annotation.model_dump())


@dataclass(frozen=True)
class SyntheticRoot:
    root: Path
    config: GenerateDatasetConfig
    manifest: ClipManifest


def make_synthetic_root(tmp_path: Path) -> SyntheticRoot:
    outputs = tmp_path / "outputs"
    root = outputs / "chat_annotation"
    manifest = write_clip(root, "src_a", "f000-006")
    publish_player_annotation(
        root,
        manifest,
        {
            1: [player("p2", [10.0, 5.0, 30.0, 50.0]), player("p1", [50.0, 10.0, 70.0, 60.0])],
            2: [player("p1", [52.0, 10.0, 72.0, 60.0]), player("p2", None)],
            4: [player("p1", [80.0, 20.0, 120.0, 70.0], truncated=True)],
        },
    )
    empty = write_clip(root, "src_b", "f010-016")
    publish_player_annotation(root, empty, {})
    raw = {
        "paths": {
            "project_root": str(tmp_path),
            "data_root": str(tmp_path / "data"),
            "checkpoint_root": str(tmp_path / "ckpt"),
            "artifact_root": str(outputs),
            "output_root": str(outputs),
            "cache_root": str(tmp_path / "cache"),
            "external_asset_root": str(tmp_path / "third_party"),
        },
        "source": {"annotation_root": "chat_annotation", "allowed_statuses": ["completed", "partial"]},
        "dataset": {"version": "test-v1", "jpeg_quality": 95},
        "split": {"val_ratio": 0.15, "test_ratio": 0.15, "seed": 0},
        "workers": 1,
        "run": {"output_dir": "player_detection/test-v1"},
    }
    return SyntheticRoot(root, GenerateDatasetConfig.from_config(OmegaConf.create(raw)), manifest)
