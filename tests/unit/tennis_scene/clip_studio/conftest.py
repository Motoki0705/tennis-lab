"""Shared fixtures for clip studio unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.project import Clip, ClipSource, ClipStudioProject
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.utils.video import VideoInfo


@pytest.fixture
def two_camera_project() -> ClipStudioProject:
    """cam0 covers global [0, 10]s, cam1 covers global [1, 9]s (offset -1)."""
    return ClipStudioProject(
        recording_id="match-001",
        sources=[
            ClipSource(path=Path("cam0.mp4"), camera_id="cam0", offset_sec=0.0),
            ClipSource(path=Path("cam1.mp4"), camera_id="cam1", offset_sec=-1.0),
        ],
        clips=[Clip(name="clip_000", start_sec=2.0, end_sec=4.0)],
    )


@pytest.fixture
def two_camera_infos() -> list[VideoInfo]:
    return [
        VideoInfo(fps=30.0, width=64, height=36, frame_count=300),
        VideoInfo(fps=30.0, width=64, height=36, frame_count=240),
    ]


@pytest.fixture
def studio_state(
    two_camera_project: ClipStudioProject, two_camera_infos: list[VideoInfo]
) -> ClipStudioState:
    return ClipStudioState(two_camera_project, two_camera_infos)
