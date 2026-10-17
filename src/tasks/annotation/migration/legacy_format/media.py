"""Check retained legacy clip timing and media identity."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from src.tasks.base.annotation.media import Timeline as Timeline
from src.tasks.base.annotation.media import decode_range as decode_range
from src.tasks.base.annotation.media import probe_video as probe_video

from .contracts import ClipManifest, sha256_file


def check_clip(video: Path, manifest: ClipManifest) -> Timeline:
    if video.name != manifest.filename or sha256_file(video) != manifest.sha256:
        raise ValueError("clip filename/hash does not match manifest")
    if video.stat().st_size != manifest.bytes:
        raise ValueError("clip byte count does not match manifest")
    timeline = probe_video(video)
    if (timeline.width, timeline.height) != (manifest.width, manifest.height):
        raise ValueError("clip dimensions do not match manifest")
    expected = [
        (
            f.clip_pts * Fraction(manifest.time_base),
            f.duration_pts * Fraction(manifest.time_base),
        )
        for f in manifest.frames
    ]
    actual = [
        (p * timeline.time_base, d * timeline.time_base)
        for p, d in zip(timeline.pts, timeline.durations, strict=True)
    ]
    if actual != expected:
        raise ValueError("clip frame count/PTS/duration does not match manifest")
    return timeline
