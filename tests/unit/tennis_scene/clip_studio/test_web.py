"""Editing transactions, stale clients and persistence errors in the browser API."""

import pytest
from pydantic import ValidationError

from src.tennis_scene.clip_studio.project import ClipStudioProject
from src.tennis_scene.clip_studio.web.service import Edit, Editor, RevisionConflict


@pytest.fixture
def editor(two_camera_project, two_camera_infos, path_resolver):
    return Editor(
        two_camera_project,
        two_camera_infos,
        path_resolver.roots.data_root / "projects.json",
        path_resolver,
    )


def test_edit_undo_redo_survives_reload(editor):
    editor.edit(Edit(revision=0, action="create", start_sec=5, end_sec=7))
    editor.edit(
        Edit(
            revision=1,
            action="update",
            name="clip_001",
            new_name="rally",
            start_sec=5.5,
            end_sec=7.5,
        )
    )
    editor.edit(Edit(revision=2, action="delete", name="clip_000"))
    assert [c.name for c in editor.project.clips] == ["rally"]
    editor.edit(Edit(revision=3, action="undo"))
    assert len(editor.project.clips) == 2
    editor.edit(Edit(revision=4, action="redo"))
    loaded = ClipStudioProject.load(
        editor.path,
        editor.resolver,
        dataset_id=editor.project.dataset_id,
        video_id=editor.project.video_id,
    )
    assert loaded.to_dict(editor.resolver) == editor.project.to_dict(editor.resolver)
    assert loaded.clips[0].start_sec == 5.5


def test_stale_browser_cannot_overwrite_new_revision(editor):
    editor.edit(Edit(revision=0, action="offsets", offsets_sec=[1, 2]))
    with pytest.raises(RevisionConflict):
        editor.edit(Edit(revision=0, action="delete", name="clip_000"))
    assert len(editor.project.clips) == 1
    assert editor.revision == 1


def test_failed_save_keeps_project_and_history(editor, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ClipStudioProject, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        editor.edit(Edit(revision=0, action="offsets", offsets_sec=[2, 3]))
    assert editor.revision == 0
    assert editor.project.sources[0].offset_sec == 0
    assert not editor.snapshot()["can_undo"]


def test_invalid_edits_do_not_mutate_or_save(editor):
    for operation in [
        Edit(revision=0, action="create", start_sec=4, end_sec=2),
        Edit(
            revision=0,
            action="update",
            name="clip_000",
            new_name="../escape",
            start_sec=2,
            end_sec=4,
        ),
        Edit(revision=0, action="offsets", offsets_sec=[1]),
    ]:
        with pytest.raises(ValueError):
            editor.edit(operation)
    assert editor.revision == 0
    assert not editor.path.exists()


def test_new_edit_after_undo_discards_redo(editor):
    editor.edit(Edit(revision=0, action="delete", name="clip_000"))
    editor.edit(Edit(revision=1, action="undo"))
    editor.edit(Edit(revision=2, action="offsets", offsets_sec=[0.5, -1]))
    assert not editor.snapshot()["can_redo"]
    assert editor.snapshot()["common"] == (1, 9)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_time_and_offsets_rejected(value):
    with pytest.raises(ValidationError):
        Edit(revision=0, action="create", start_sec=value, end_sec=3)
    with pytest.raises(ValidationError):
        Edit(revision=0, action="offsets", offsets_sec=[value])


def test_sync_is_a_proposal_and_cancel_discards_it(editor, monkeypatch, tmp_path):
    import threading
    from time import monotonic, sleep

    from src.tennis_scene.clip_studio.audio_sync import AudioSyncResult
    from src.tennis_scene.clip_studio.export import ExportSettings
    from src.tennis_scene.clip_studio.web.jobs import JobRequest, Jobs
    from src.tennis_scene.configuration import AudioSyncRuntimeConfig

    entered = threading.Event()
    release = threading.Event()

    def estimate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return AudioSyncResult([0, 2], [1, 0.9], 0)

    monkeypatch.setattr(
        "src.tennis_scene.clip_studio.web.jobs.estimate_audio_offsets", estimate
    )
    jobs = Jobs(
        editor,
        ExportSettings(tmp_path / "dataset", 30, 64, 36, 17, False),
        AudioSyncRuntimeConfig(8000, 100, None),
    )
    try:
        jobs.start(JobRequest(revision=0, kind="sync"))
        assert entered.wait(5)
        with pytest.raises(ValueError, match="処理中"):
            jobs.start(JobRequest(revision=0, kind="sync"))
        jobs.cancel()
        release.set()
        deadline = monotonic() + 5
        while jobs.snapshot()["status"] == "running" and monotonic() < deadline:
            sleep(0.01)
        assert jobs.snapshot()["status"] == "cancelled"
        assert "offsets_sec" not in jobs.snapshot()
        assert editor.revision == 0
        assert editor.project.sources[1].offset_sec == -1
        jobs.start(JobRequest(revision=0, kind="sync"))
        deadline = monotonic() + 5
        while jobs.snapshot()["status"] == "running" and monotonic() < deadline:
            sleep(0.01)
        assert jobs.snapshot()["status"] == "done"
        assert jobs.snapshot()["offsets_sec"] == [0, 2]
        assert editor.project.sources[1].offset_sec == -1
    finally:
        release.set()
        jobs.close()
