"""Tests for src/tennis_scene/clip_studio/project.py."""

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipSource,
    ClipStudioProject,
)
from src.utils.io import load_json, save_json_atomic


class TestValidate:
    def test_valid_project(self, two_camera_project: ClipStudioProject) -> None:
        assert two_camera_project.validate() == []

    def test_empty_sources(self) -> None:
        errors = ClipStudioProject().validate()
        assert any("at least one source" in error for error in errors)

    def test_recording_id_is_required(self) -> None:
        errors = ClipStudioProject(
            sources=[ClipSource(path=Path("a.mp4"), camera_id="cam0")]
        ).validate()
        assert any("recording_id" in error for error in errors)

    @pytest.mark.parametrize(
        "recording_id", ["../match", "match/one", "match one", "match:one"]
    )
    def test_recording_id_must_be_safe_path_component(
        self, two_camera_project: ClipStudioProject, recording_id: str
    ) -> None:
        two_camera_project.recording_id = recording_id
        assert any("recording_id" in error for error in two_camera_project.validate())

    def test_duplicate_camera_ids(self) -> None:
        project = ClipStudioProject(
            sources=[
                ClipSource(path=Path("a.mp4"), camera_id="cam0"),
                ClipSource(path=Path("b.mp4"), camera_id="cam0"),
            ]
        )
        assert any("unique" in error for error in project.validate())

    def test_non_finite_offset(self) -> None:
        project = ClipStudioProject(
            sources=[
                ClipSource(path=Path("a.mp4"), camera_id="cam0", offset_sec=float("nan"))
            ]
        )
        assert any("finite" in error for error in project.validate())

    def test_bad_clip_bounds(self, two_camera_project: ClipStudioProject) -> None:
        two_camera_project.clips.append(Clip(name="bad", start_sec=5.0, end_sec=5.0))
        assert any("end_sec > start_sec" in error for error in two_camera_project.validate())

    def test_duplicate_clip_names(self, two_camera_project: ClipStudioProject) -> None:
        two_camera_project.clips.append(Clip(name="clip_000", start_sec=5.0, end_sec=6.0))
        assert any("unique" in error for error in two_camera_project.validate())

    def test_clip_name_must_be_safe_path_component(
        self, two_camera_project: ClipStudioProject
    ) -> None:
        two_camera_project.clips[0].name = "../escape"
        assert any("clip name" in error for error in two_camera_project.validate())


class TestNaming:
    def test_next_clip_name_skips_used(self, two_camera_project: ClipStudioProject) -> None:
        assert two_camera_project.next_clip_name() == "clip_001"
        two_camera_project.clips.append(Clip(name="clip_001", start_sec=5.0, end_sec=6.0))
        assert two_camera_project.next_clip_name() == "clip_002"

    def test_clip_index_by_name(self, two_camera_project: ClipStudioProject) -> None:
        assert two_camera_project.clip_index_by_name("clip_000") == 0
        with pytest.raises(KeyError, match="not found"):
            two_camera_project.clip_index_by_name("nope")


class TestPersistence:
    def test_round_trip(
        self, two_camera_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        project = two_camera_project
        project.sources[0].path = tmp_path / "cam0.mp4"
        project.sources[1].path = tmp_path / "cam1.mp4"
        path = tmp_path / "project.json"
        project.save(path)

        loaded = ClipStudioProject.load(path)
        assert [source.to_dict() for source in loaded.sources] == [
            source.to_dict() for source in project.sources
        ]
        assert [clip.to_dict() for clip in loaded.clips] == [
            clip.to_dict() for clip in project.clips
        ]

    def test_relative_paths_resolved_against_project_dir(self, tmp_path: Path) -> None:
        data = {
            "version": 2,
            "recording_id": "match-001",
            "sources": [
                {"path": "videos/cam0.mp4", "camera_id": "cam0", "offset_sec": 0.0}
            ],
            "clips": [],
        }
        path = tmp_path / "nested" / "project.json"
        save_json_atomic(data, path)
        loaded = ClipStudioProject.load(path)
        assert loaded.sources[0].path == (tmp_path / "nested" / "videos/cam0.mp4").resolve()

    def test_save_invalid_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid project"):
            ClipStudioProject().save(tmp_path / "bad.json")

    def test_version_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "old.json"
        save_json_atomic({"version": 0, "sources": [], "clips": []}, path)
        with pytest.raises(ValueError, match="Unsupported project version"):
            ClipStudioProject.load(path)

    def test_saved_json_is_plain_data(
        self, two_camera_project: ClipStudioProject, tmp_path: Path
    ) -> None:
        path = tmp_path / "project.json"
        two_camera_project.save(path)
        data = load_json(path)
        assert data["version"] == 2
        assert data["recording_id"] == "match-001"
        assert data["sources"][0]["camera_id"] == "cam0"
        assert data["clips"][0]["name"] == "clip_000"
