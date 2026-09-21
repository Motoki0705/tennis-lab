from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.runtime.contracts import (
    Annotation,
    ClipManifest,
    FrameMap,
    FrameRange,
    Policies,
    SourceInfo,
    make_template,
    read_json,
)


@pytest.fixture(scope="session")
def kit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    return build_kit(tmp_path_factory.mktemp("project_kits"))


@pytest.fixture
def definition(kit: tuple[Path, str]) -> dict[str, Any]:
    value: dict[str, Any] = read_json(kit[0] / "court_definition.json")
    return value


@pytest.fixture
def manifest(kit: tuple[Path, str]) -> ClipManifest:
    return ClipManifest(
        schema_version="tennis_chat_clip.v1",
        kit_version="1.0.0",
        kit_id=kit[1],
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
            static_tolerance_px_at_1080p=3,
            homography_max_error_px_at_1080p=3,
            ball_max_gap_seconds=0.1,
        ),
    )


@pytest.fixture
def annotation(manifest: ClipManifest) -> Annotation:
    result = make_template(manifest, "c" * 64)
    result.court_mode = "unavailable"
    result.inspection_ranges = [FrameRange(start=0, stop=12)]
    for frame in result.frames:
        frame.people_review = frame.balls_review = frame.court_review = "complete"
    return result
