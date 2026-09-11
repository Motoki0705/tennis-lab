"""Initial sync is all-or-nothing; saved annotations are authoritative."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from src.tennis_scene.clip_studio.initialization import (
    initialize_recording_offsets,
    load_or_create_project,
)
from src.tennis_scene.clip_studio.project import Clip, ClipSource, ClipStudioProject
from src.tennis_scene.configuration import parse_clip_studio_config
from src.utils.video.metadata import CreationTimeError
from tests.unit.tennis_scene.test_configuration import _clip_studio_config


@pytest.fixture
def runtime(tmp_path):
    value = parse_clip_studio_config(OmegaConf.create(_clip_studio_config(tmp_path)))
    assert value.video_paths is not None
    for path in value.video_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return value


@pytest.mark.parametrize(
    "deltas, expected",
    [
        ([0, 7, 12], [12, 5, 0]),
        ([12, 0, 7], [0, 12, 5]),
        ([1, 1, 1], [0, 0, 0]),
        ([0.125, 0.5], [0.375, 0]),
        ([0], [0]),
    ],
)
def test_latest_recording_is_global_zero(tmp_path, deltas, expected):
    project = ClipStudioProject(
        dataset_id="test",
        video_id="video_000",
        sources=[
            ClipSource(tmp_path / f"{i}.mp4", f"cam{i}") for i in range(len(deltas))
        ],
    )
    start = datetime(2026, 7, 9, tzinfo=UTC)
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=[start + timedelta(seconds=d) for d in deltas],
    ):
        notice = initialize_recording_offsets(project)
    assert [s.offset_sec for s in project.sources] == expected
    assert not notice.warning


def test_initial_values_saved_and_manual_edits_survive_restart(runtime):
    start = datetime(2026, 7, 9, tzinfo=UTC)
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=[start, start + timedelta(seconds=12)],
    ):
        project, notice = load_or_create_project(runtime)
    assert notice and not notice.warning
    assert [
        s.offset_sec
        for s in ClipStudioProject.load(
            runtime.export.projects_path,
            runtime.export.resolver,
            dataset_id=runtime.dataset_id,
            video_id=runtime.video_id,
        ).sources
    ] == [12, 0]
    project.sources[0].offset_sec = 11.75
    project.clips.append(Clip("rally", 1, 2))
    project.save(runtime.export.projects_path, runtime.export.resolver)
    before = runtime.export.projects_path.read_bytes()
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=AssertionError("Must not probe existing projects"),
    ):
        resumed, notice = load_or_create_project(runtime)
    assert notice is None
    assert resumed.sources[0].offset_sec == 11.75
    assert resumed.clips == project.clips
    assert runtime.export.projects_path.read_bytes() == before


def test_failed_initialization_not_retried_even_with_zero_offsets_and_no_clips(
    runtime, caplog
):
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=[
            datetime(2026, 7, 9, tzinfo=UTC),
            CreationTimeError("creation_time がありません"),
        ],
    ):
        project, notice = load_or_create_project(runtime)
    assert notice and notice.warning
    assert "cam1" in notice.message
    assert "creation_time がありません" in caplog.text
    assert [s.offset_sec for s in project.sources] == [0, 0]
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=AssertionError("Must not retry"),
    ):
        resumed, notice = load_or_create_project(runtime)
    assert notice is None
    assert resumed == project


def test_all_failed_cameras_reported(runtime):
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=CreationTimeError("不正"),
    ):
        _, notice = load_or_create_project(runtime)
    assert notice is not None
    assert "cam0" in notice.message and "cam1" in notice.message


def test_existing_project_rejects_discovered_source_drift(runtime, tmp_path):
    with patch(
        "src.tennis_scene.clip_studio.initialization.read_creation_time",
        side_effect=CreationTimeError("missing"),
    ):
        load_or_create_project(runtime)
    other = tmp_path / "data/other.mp4"
    other.touch()
    with pytest.raises(ValueError, match="differ from the discovered"):
        load_or_create_project(
            replace(runtime, video_paths=(other, *runtime.video_paths[1:]))
        )
