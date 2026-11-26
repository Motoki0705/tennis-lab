"""Tests for src/tennis_scene/clip_studio/project.py."""

from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipScore,
    ClipSource,
    ClipStudioProject,
    ClipStudioProjects,
    MatchSettings,
)
from src.utils.configuration import PathContractError, PathResolver, PathRole
from src.utils.io import load_json, save_json_atomic
from src.utils.tennis_scoring import ScoreCount, SoftTennisRules


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
        project.match = MatchSettings("青ペア", "赤ペア", SoftTennisRules(7), "B")
        project.clips[0].score = ClipScore(
            "confirmed",
            ScoreCount(games=(1, 2), points=(2, 1)),
            "anchor",
            "A",
            "auto",
            "B",
            "映像で確認",
        )
        project.clips.append(
            Clip(
                "pause",
                4,
                5,
                ClipScore(
                    "not_point",
                    None,
                    "anchor",
                    "A",
                    "manual",
                    None,
                    "中断",
                ),
            )
        )
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
        assert loaded.match == project.match
        raw = load_json(path)["projects"]["video_000"]["clips"]
        assert raw[0]["score"]["before"] == {"games": [1, 2], "points": [2, 1]}
        assert "before" not in raw[1]["score"]

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
                    "clips": [{"name": "rally", "start_sec": 1.2, "end_sec": 3.4}],
                }
            },
        }
        path = path_resolver.resolve(PathRole.DATA, "nested/project.json")
        save_json_atomic(data, path)
        original = path.read_bytes()
        loaded = ClipStudioProject.load(
            path,
            path_resolver,
            dataset_id="test-dataset",
            video_id="video_000",
        )
        assert loaded.sources[0].path == path_resolver.resolve(
            PathRole.DATA, "videos/cam0.mp4"
        )
        assert loaded.clips[0].score is None and loaded.match is None
        assert path.read_bytes() == original
        loaded.save(path, path_resolver)
        saved = load_json(path)
        assert saved["version"] == 2
        assert saved["projects"]["video_000"]["clips"][0] == {
            "name": "rally",
            "start_sec": 1.2,
            "end_sec": 3.4,
            "score": None,
        }
        assert saved["projects"]["video_000"]["match"] is None

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
        assert data["version"] == 2
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


@pytest.mark.parametrize(
    "score_change,message",
    [
        ({"before": {"games": [4, 0], "points": [0, 0]}}, "試合終了後"),
        ({"before": {"games": [0, 0], "points": [True, 0]}}, "非負整数"),
        ({"before": {"games": [0, 0], "points": [1, 0], "sets": [0, 0]}}, "硬式でのみ"),
        ({"status": "confirmed"}, "得点側"),
        ({"status": "unknown", "note": "  "}, "理由"),
        ({"status": "multiple_points", "winner": "A"}, "得点側"),
        ({"status": "not_point"}, "keys must be exactly"),
        ({"unexpected": 1}, "keys must be exactly"),
    ],
)
def test_v2_rejects_invalid_scores_with_clip_name(
    two_camera_project: ClipStudioProject,
    path_resolver: PathResolver,
    score_change: dict[str, Any],
    message: str,
) -> None:
    video = two_camera_project.to_dict(path_resolver)
    video["match"] = MatchSettings("A", "B", SoftTennisRules(7), "A").to_dict()
    video["clips"][0]["score"] = {
        "status": "auto",
        "before": {"games": [0, 0], "points": [0, 0]},
        "before_source": "anchor",
        "server": "A",
        "server_source": "auto",
        "winner": None,
        "note": "",
        **score_change,
    }
    with pytest.raises(ValueError, match=f"clip 'clip_000'.*{message}"):
        ClipStudioProjects.from_dict(
            {
                "version": 2,
                "dataset_id": "test-dataset",
                "projects": {"video_000": video},
            },
            path_resolver,
        )


@pytest.mark.parametrize("missing", ["match", "score"])
def test_v2_requires_version_specific_keys(
    two_camera_project: ClipStudioProject, path_resolver: PathResolver, missing: str
) -> None:
    video = two_camera_project.to_dict(path_resolver)
    target = video if missing == "match" else video["clips"][0]
    del target[missing]
    with pytest.raises(ValueError, match="keys must be exactly"):
        ClipStudioProjects.from_dict(
            {
                "version": 2,
                "dataset_id": "test-dataset",
                "projects": {"video_000": video},
            },
            path_resolver,
        )


@pytest.mark.parametrize(
    "rules",
    [
        {"type": "soft_tennis"},
        {"type": "soft_tennis", "match_games": 6},
        {"type": "tennis", "best_of": 3, "no_ad": "false", "final_set": "tie_break"},
        {"type": "tennis", "best_of": 3, "no_ad": False, "final_set": "unknown"},
        {"type": "unknown"},
    ],
)
def test_match_rejects_invalid_rule_settings(rules: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MatchSettings.from_dict(
            {"sides": {"A": "A", "B": "B"}, "first_server": "A", "rules": rules}
        )
