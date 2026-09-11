"""Tests for src/tennis_scene/clip_studio/project.py."""

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipSource,
    ClipStudioProject,
    ClipStudioProjects,
)
from src.utils.configuration import PathContractError, PathResolver, PathRole
from src.utils.io import load_json, save_json_atomic


class TestValidate:
    def test_valid_project(self, two_camera_project: ClipStudioProject) -> None:
        assert two_camera_project.validate() == []

    def test_empty_sources(self) -> None:
        errors = ClipStudioProject().validate()
        assert any("at least one source" in error for error in errors)

    def test_dataset_and_video_ids_are_required(self) -> None:
        errors = ClipStudioProject(
            sources=[ClipSource(path=Path("a.mp4"), camera_id="cam0")]
        ).validate()
        assert any("dataset_id" in error for error in errors)
        assert any("video_id" in error for error in errors)

    @pytest.mark.parametrize(
        "video_id", ["../video", "video/one", "video one", "video:one"]
    )
    def test_video_id_must_be_safe_path_component(
        self, two_camera_project: ClipStudioProject, video_id: str
    ) -> None:
        two_camera_project.video_id = video_id
        assert any("video_id" in error for error in two_camera_project.validate())

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
                ClipSource(
                    path=Path("a.mp4"), camera_id="cam0", offset_sec=float("nan")
                )
            ]
        )
        assert any("finite" in error for error in project.validate())

    def test_bad_clip_bounds(self, two_camera_project: ClipStudioProject) -> None:
        two_camera_project.clips.append(Clip(name="bad", start_sec=5.0, end_sec=5.0))
        assert any(
            "end_sec > start_sec" in error for error in two_camera_project.validate()
        )

    def test_duplicate_clip_names(self, two_camera_project: ClipStudioProject) -> None:
        two_camera_project.clips.append(
            Clip(name="clip_000", start_sec=5.0, end_sec=6.0)
        )
        assert any("unique" in error for error in two_camera_project.validate())

    def test_clip_name_must_be_safe_path_component(
        self, two_camera_project: ClipStudioProject
    ) -> None:
        two_camera_project.clips[0].name = "../escape"
        assert any("clip name" in error for error in two_camera_project.validate())


class TestNaming:
    def test_next_clip_name_skips_used(
        self, two_camera_project: ClipStudioProject
    ) -> None:
        assert two_camera_project.next_clip_name() == "clip_001"
        two_camera_project.clips.append(
            Clip(name="clip_001", start_sec=5.0, end_sec=6.0)
        )
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
        path = path_resolver.resolve(PathRole.DATA, "project.json")
        project.save(path, path_resolver)

        loaded = ClipStudioProject.load(
            path,
            path_resolver,
            dataset_id=project.dataset_id,
            video_id=project.video_id,
        )
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
            "version": 1,
            "dataset_id": "test-dataset",
            "projects": {
                "video_000": {
                    "sources": [
                        {
                            "path": "videos/cam0.mp4",
                            "camera_id": "cam0",
                            "offset_sec": 0.0,
                        }
                    ],
                    "clips": [],
                }
            },
        }
        path = path_resolver.resolve(PathRole.DATA, "nested/project.json")
        save_json_atomic(data, path)
        loaded = ClipStudioProject.load(
            path,
            path_resolver,
            dataset_id="test-dataset",
            video_id="video_000",
        )
        assert loaded.sources[0].path == path_resolver.resolve(
            PathRole.DATA, "videos/cam0.mp4"
        )

    def test_absolute_source_path_is_rejected(
        self, path_resolver: PathResolver
    ) -> None:
        data = {
            "version": 1,
            "dataset_id": "test-dataset",
            "projects": {
                "video_000": {
                    "sources": [
                        {"path": "/etc/passwd", "camera_id": "cam0", "offset_sec": 0.0}
                    ],
                    "clips": [],
                }
            },
        }
        path = path_resolver.resolve(PathRole.DATA, "absolute.json")
        save_json_atomic(data, path)
        with pytest.raises(PathContractError, match="must be relative"):
            ClipStudioProject.load(
                path,
                path_resolver,
                dataset_id="test-dataset",
                video_id="video_000",
            )

    def test_save_invalid_raises(self, path_resolver: PathResolver) -> None:
        with pytest.raises(ValueError, match="Invalid project"):
            ClipStudioProject().save(
                path_resolver.resolve(PathRole.DATA, "bad.json"),
                path_resolver,
            )

    def test_version_mismatch_raises(self, path_resolver: PathResolver) -> None:
        path = path_resolver.resolve(PathRole.DATA, "old.json")
        save_json_atomic({"version": 0, "dataset_id": "test", "projects": {}}, path)
        with pytest.raises(ValueError, match="Unsupported projects version"):
            ClipStudioProject.load(
                path,
                path_resolver,
                dataset_id="test",
                video_id="video_000",
            )

    def test_saved_json_is_plain_data(
        self,
        two_camera_project: ClipStudioProject,
        path_resolver: PathResolver,
    ) -> None:
        path = path_resolver.resolve(PathRole.DATA, "project.json")
        two_camera_project.save(path, path_resolver)
        data = load_json(path)
        assert data["version"] == 1
        assert data["dataset_id"] == "test-dataset"
        assert data["projects"]["video_000"]["sources"][0]["camera_id"] == "cam0"
        assert data["projects"]["video_000"]["clips"][0]["name"] == "clip_000"

    def test_saving_another_video_preserves_existing_project(
        self,
        two_camera_project: ClipStudioProject,
        path_resolver: PathResolver,
    ) -> None:
        path = path_resolver.resolve(PathRole.DATA, "projects.json")
        two_camera_project.save(path, path_resolver)
        second = ClipStudioProject(
            dataset_id=two_camera_project.dataset_id,
            video_id="video_001",
            sources=[two_camera_project.sources[0]],
        )
        second.save(path, path_resolver)

        projects = ClipStudioProjects.load(path, path_resolver)
        assert sorted(projects.projects) == ["video_000", "video_001"]
        assert projects.projects["video_000"].clips[0].name == "clip_000"
