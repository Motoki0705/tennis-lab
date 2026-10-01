"""Bind a clip's explicit media and timeline before any component executes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.utils.checksum import dual_sha256
from src.utils.video import probe_video_info


def build_clip_source(paths: Sequence[Path], camera_ids: Sequence[str], *, max_frames: int | None = None,
                      clip_id: str | None = None) -> ClipSource:
    if not 3 <= len(paths) <= 5 or len(paths) != len(camera_ids) or len(set(camera_ids)) != len(paths):
        raise ValueError("Scene reconstruction requires 3..5 unique cameras")
    videos: list[SourceVideo] = []
    for path, camera_id in zip(paths, camera_ids, strict=True):
        info = probe_video_info(path)
        frames = info.frame_count if max_frames is None else min(info.frame_count, max_frames)
        if info.fps < 28 or frames < 1:
            raise ValueError("Scene reconstruction requires nonempty video at >=28 FPS")
        videos.append(SourceVideo(camera_id, path.resolve(), dual_sha256(path), frames, info.fps, info.width, info.height))
    return ClipSource(clip_id or "standalone", tuple(videos))
