"""Bind task-owned input identities to original video bytes and native decoding."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from src.tasks.base.annotation.media import Timeline, probe_video
from src.tasks.base.annotation.plan import file_digest

from .contracts import VideoAnnotationInput


def validate_video_input(path: Path, expected: VideoAnnotationInput) -> Timeline:
    expected = VideoAnnotationInput.model_validate_json(expected.canonical_bytes())
    source = expected.source
    if (
        not path.is_file()
        or path.stat().st_size != source.size_bytes
        or file_digest(path) != source.media_sha256
    ):
        raise ValueError("original video bytes differ from VideoAnnotationInput")
    timeline = probe_video(path)
    if (
        (timeline.width, timeline.height, len(timeline.pts))
        != (source.width, source.height, source.frame_count)
        or timeline.time_base != Fraction(source.time_base)
        or timeline.rate != Fraction(source.nominal_fps)
    ):
        raise ValueError("original video metadata differs from VideoAnnotationInput")
    if any(timeline.pts[row.frame_index] != row.pts for row in expected.frames):
        raise ValueError("original video frame/PTS differs from VideoAnnotationInput")
    return timeline
