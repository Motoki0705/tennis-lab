"""Shared fixtures for clip studio unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.project import Clip, ClipSource, ClipStudioProject
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.video import VideoInfo


@pytest.fixture
def path_resolver(tmp_path: Path) -> PathResolver:
    roots = RuntimePathRoots(
        project_root=tmp_path / "project",
        data_root=tmp_path / "data",
        checkpoint_root=tmp_path / "checkpoint",
        artifact_root=tmp_path / "artifact",
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        external_asset_root=tmp_path / "external",
    )
    for path in roots.as_mapping().values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return PathResolver(roots)


@pytest.fixture
def two_camera_project(path_resolver: PathResolver) -> ClipStudioProject:
    """cam0 covers global [0, 10]s, cam1 covers global [1, 9]s (offset -1)."""
    return ClipStudioProject(
        recording_id="match-001",
        sources=[
            ClipSource(
                path=path_resolver.resolve(PathRole.DATA, "cam0.mp4"),
                camera_id="cam0",
                offset_sec=0.0,
            ),
            ClipSource(
                path=path_resolver.resolve(PathRole.DATA, "cam1.mp4"),
                camera_id="cam1",
                offset_sec=-1.0,
            ),
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
