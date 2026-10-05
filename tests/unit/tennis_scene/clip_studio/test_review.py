"""Authoritative time mapping and truthful, read-only export inventory."""

from copy import deepcopy

import pytest

from src.tennis_scene.clip_studio.export import (
    ExportSettings,
    build_clip_manifest,
    plan_clip_export,
)
from src.tennis_scene.clip_studio.project import Clip
from src.tennis_scene.clip_studio.web.review import frame_correspondence, review_catalog
from src.tennis_scene.clip_studio.web.service import Edit, Editor
from src.tennis_scene.generate_dataset.manifest import (
    DatasetManifest,
    register_exported_clip,
)
from src.utils.io import load_json, save_json_atomic


@pytest.fixture
def review_data(two_camera_project, two_camera_infos, path_resolver):
    project = two_camera_project
    infos = two_camera_infos
    projects = path_resolver.roots.data_root / "projects.json"
    project.save(projects, path_resolver)
    output = path_resolver.roots.data_root / "dataset"
    directory = output / "videos/video_000/clips/clip_000"
    settings = ExportSettings(output, 30, 64, 36, 17, False)
    plan = plan_clip_export(project, infos, project.clips[0], settings)
    save_json_atomic(build_clip_manifest(plan), directory / "clip.json")
    for camera in project.sources:
        path = directory / "media" / f"{camera.camera_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not decoded: catalog only checks file existence")
    register_exported_clip(output, directory / "clip.json")
    return project, infos, projects, output, directory


def test_time_mapping_preserves_missing_frames_and_half_open_clips(
    two_camera_project, two_camera_infos
):
    result = frame_correspondence(two_camera_project, two_camera_infos, 2)
    assert [c["frame_index"] for c in result["cameras"]] == [60, 30]
    assert [c["local_time_sec"] for c in result["cameras"]] == [2, 1]
    assert result["containing_clips"] == ["clip_000"]
    assert (
        frame_correspondence(two_camera_project, two_camera_infos, 4)[
            "containing_clips"
        ]
        == []
    )
    before = frame_correspondence(two_camera_project, two_camera_infos, 0)
    assert before["cameras"][1]["frame_index"] is None
    assert not before["cameras"][1]["available"]
    # Python's ties-to-even rounding is shared with the actual JPEG endpoint.
    assert (
        frame_correspondence(two_camera_project, two_camera_infos, 0.5 / 30)["cameras"][
            0
        ]["frame_index"]
        == 0
    )


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -float("inf")])
def test_time_mapping_rejects_nonfinite_time(
    two_camera_project, two_camera_infos, time
):
    with pytest.raises(ValueError, match="有限"):
        frame_correspondence(two_camera_project, two_camera_infos, time)


def test_registered_export_and_pending_clip_are_distinct_without_writes(review_data):
    project, infos, projects, output, directory = review_data
    project.clips.append(Clip("clip_001", 5, 6))
    before = {
        p: p.read_bytes()
        for p in (projects, output / "dataset.json", directory / "clip.json")
    }
    result = review_catalog(project, infos, projects, output)
    assert result["counts"] == {"indexed": 1, "unexported": 1}
    assert result["video_registered_total"] == 1
    assert result["clips"][0]["export"]["num_frames"] == 60
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (output / "videos/video_000/clips/clip_001").exists()


def test_missing_media_is_not_reported_as_registered(review_data):
    project, infos, projects, output, directory = review_data
    (directory / "media/cam1.mp4").unlink()
    clip = review_catalog(project, infos, projects, output)["clips"][0]
    assert clip["state"] == "incomplete"
    assert "cam1" in clip["reason"]


@pytest.mark.parametrize("change", ["offset", "bounds", "index"])
def test_saved_project_and_registered_record_conflicts_are_visible(review_data, change):
    project, infos, projects, output, directory = review_data
    if change == "offset":
        project.sources[1].offset_sec += 0.1
    elif change == "bounds":
        project.clips[0].start_sec += 0.1
    else:
        index = load_json(output / "dataset.json")
        index["clips"][0]["num_frames"] += 1
        save_json_atomic(index, output / "dataset.json")
    result = review_catalog(project, infos, projects, output)["clips"][0]
    assert result["state"] == "conflict"
    assert "不一致" in result["reason"]


def test_unknown_index_and_missing_registration_are_distinct(review_data):
    project, infos, projects, output, directory = review_data
    DatasetManifest(dataset_id=project.dataset_id).save(output)
    assert (
        review_catalog(project, infos, projects, output)["clips"][0]["state"]
        == "unregistered"
    )
    (output / "dataset.json").write_text("{invalid")
    result = review_catalog(project, infos, projects, output)
    assert result["index_error"]
    assert result["clips"][0]["state"] == "index_unknown"
    assert result["video_registered_total"] is None


def test_old_or_malformed_manifests_have_explicit_failure(review_data):
    project, infos, projects, output, directory = review_data
    (directory / "clip.json").write_text('{"version": 1}')
    result = review_catalog(project, infos, projects, output)["clips"][0]
    assert result["state"] == "invalid"
    assert "確認できません" in result["reason"]


@pytest.mark.parametrize(
    "action", ["create", "update", "delete", "offsets", "undo", "redo"]
)
def test_read_only_editor_rejects_all_edits_before_state_or_file_changes(
    review_data, path_resolver, action
):
    project, infos, projects, output, directory = review_data
    editor = Editor(project, infos, projects, path_resolver, read_only=True)
    original = projects.read_bytes()
    state = deepcopy(editor.snapshot())
    with pytest.raises(PermissionError, match="読取専用"):
        editor.edit(Edit(revision=0, action=action))
    assert editor.snapshot() == state
    assert projects.read_bytes() == original
