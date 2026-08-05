"""Tests for src/tennis_scene/clip_studio/project.py."""

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipSource,
    ClipStudioProject,
)
from src.utils.configuration import PathContractError, PathResolver, PathRole
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
        self,
        two_camera_project: ClipStudioProject,
        path_resolver: PathResolver,
    ) -> None:
        project = two_camera_project
        path = path_resolver.resolve(PathRole.ARTIFACT, "project.json")
        project.save(path, path_resolver)

        loaded = ClipStudioProject.load(path, path_resolver)
        assert [source.to_dict(path_resolver) for source in loaded.sources] == [
            source.to_dict(path_resolver) for source in project.sources
        ]
        assert [clip.to_dict() for clip in loaded.clips] == [
            clip.to_dict() for clip in project.clips
        ]

    def test_relative_paths_resolved_against_data_root(
        self, path_resolver: PathResolver
    ) -> None:
        data = {
            "version": 2,
            "recording_id": "match-001",
            "sources": [
                {"path": "videos/cam0.mp4", "camera_id": "cam0", "offset_sec": 0.0}
            ],
            "clips": [],
        }
        path = path_resolver.resolve(PathRole.ARTIFACT, "nested/project.json")
        save_json_atomic(data, path)
        loaded = ClipStudioProject.load(path, path_resolver)
        assert loaded.sources[0].path == path_resolver.resolve(
            PathRole.DATA, "videos/cam0.mp4"
        )

    def test_absolute_source_path_is_rejected(
        self, path_resolver: PathResolver
    ) -> None:
        data = {
            "version": 2,
            "recording_id": "match-001",
            "sources": [
                {"path": "/etc/passwd", "camera_id": "cam0", "offset_sec": 0.0}
            ],
            "clips": [],
        }
        path = path_resolver.resolve(PathRole.ARTIFACT, "absolute.json")
        save_json_atomic(data, path)
        with pytest.raises(PathContractError, match="must be relative"):
            ClipStudioProject.load(path, path_resolver)

    def test_save_invalid_raises(self, path_resolver: PathResolver) -> None:
        with pytest.raises(ValueError, match="Invalid project"):
            ClipStudioProject().save(
                path_resolver.resolve(PathRole.ARTIFACT, "bad.json"),
                path_resolver,
            )

    def test_version_mismatch_raises(self, path_resolver: PathResolver) -> None:
        path = path_resolver.resolve(PathRole.ARTIFACT, "old.json")
        save_json_atomic({"version": 0, "sources": [], "clips": []}, path)
        with pytest.raises(ValueError, match="Unsupported project version"):
            ClipStudioProject.load(path, path_resolver)

    def test_saved_json_is_plain_data(
        self,
        two_camera_project: ClipStudioProject,
        path_resolver: PathResolver,
    ) -> None:
        path = path_resolver.resolve(PathRole.ARTIFACT, "project.json")
        two_camera_project.save(path, path_resolver)
        data = load_json(path)
        assert data["version"] == 2
        assert data["recording_id"] == "match-001"
        assert data["sources"][0]["camera_id"] == "cam0"
        assert data["clips"][0]["name"] == "clip_000"
