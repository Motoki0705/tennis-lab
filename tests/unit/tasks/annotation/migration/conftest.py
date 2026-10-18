from __future__ import annotations

import pytest

from src.tasks.annotation.migration.legacy_format.contracts import (
    Annotation,
    ClipManifest,
    FrameMap,
    FrameRange,
    Policies,
    SourceInfo,
)
from tests.support.legacy_annotations import KIT_VERSION, make_template


@pytest.fixture
def manifest() -> ClipManifest:
    return ClipManifest(
        schema_version="tennis_chat_clip.v1",
        kit_version=KIT_VERSION,
        kit_id="c" * 64,
        clip_id="clip_000",
        source=SourceInfo(
            source_id="sample",
            youtube_id=None,
            url=None,
            title="Synthetic fixture",
            filename="source.mp4",
            sha256="a" * 64,
            bytes=1000,
            acquired_at="2026-09-21",
        ),
        filename="sample.mp4",
        sha256="b" * 64,
        bytes=1000,
        width=1920,
        height=1080,
        time_base="1/30",
        nominal_fps="30",
        source_start_pts=0,
        media_range=FrameRange(start=0, stop=12),
        target_range=FrameRange(start=0, stop=12),
        frames=[
            FrameMap(
                frame_index=i,
                source_frame_index=i,
                source_pts=i,
                clip_pts=i,
                duration_pts=1,
                is_target=True,
            )
            for i in range(12)
        ],
        policies=Policies(
            ball_max_gap_seconds=0.1,
        ),
    )


@pytest.fixture
def annotation(manifest: ClipManifest) -> Annotation:
    result = make_template(manifest)
    result.status = "completed"
    for frame in result.frames:
        frame.reviewed = True
    return result
