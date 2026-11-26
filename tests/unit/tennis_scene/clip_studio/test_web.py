"""Editing transactions, stale clients and persistence errors in the browser API."""

import pytest
from pydantic import ValidationError

from src.tennis_scene.clip_studio.project import ClipStudioProject
from src.tennis_scene.clip_studio.score_csv import export_scores
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


def _match_payload(*, no_ad=False):
    return {
        "sides": {"A": "青", "B": "赤"},
        "rules": {
            "type": "tennis",
            "best_of": 3,
            "no_ad": no_ad,
            "final_set": "tie_break",
        },
        "first_server": "A",
    }


def _anchor_winner(*, before, winner):
    return {
        "status": "confirmed",
        "before": before,
        "before_source": "anchor",
        "server": None,
        "server_source": "auto",
        "winner": winner,
        "note": "",
    }


def _loaded(editor):
    return ClipStudioProject.load(
        editor.path,
        editor.resolver,
        dataset_id=editor.project.dataset_id,
        video_id=editor.project.video_id,
    )


def test_score_and_rules_preview_save_and_undo_as_single_operations(editor):
    editor.edit(Edit(revision=0, action="create", start_sec=5, end_sec=7))
    editor.edit(Edit(revision=1, action="set_match", match=_match_payload()))
    score = _anchor_winner(
        before={"sets": [0, 0], "games": [0, 0], "points": [3, 3]}, winner="B"
    )
    operation = Edit(revision=2, action="set_score", name="clip_000", score=score)
    before = editor.path.read_bytes()
    assert editor.preview(operation) == {
        "revision": 2,
        "changed_clips": 1,
        "broken_chain_clips": [],
    }
    assert editor.path.read_bytes() == before
    assert all(clip.score is None for clip in editor.project.clips)
    scored = editor.edit(operation)
    assert scored["clips"][1]["score"]["before"]["points"] == [3, 4]
    assert scored["score_views"]["clip_000"]["after"]["points_label"] == "AD B"
    with pytest.raises(RevisionConflict):
        editor.edit(operation)
    with pytest.raises(RevisionConflict):
        editor.preview(operation)

    rule_edit = Edit(revision=3, action="set_match", match=_match_payload(no_ad=True))
    assert editor.preview(rule_edit) == {
        "revision": 3,
        "changed_clips": 1,
        "broken_chain_clips": [],
    }
    updated = editor.edit(rule_edit)
    assert updated["clips"][1]["score"]["before"] == {
        "sets": [0, 0],
        "games": [0, 1],
        "points": [0, 0],
    }
    assert updated["clips"][1]["score"]["server"] == "B"
    assert _loaded(editor).clips == editor.project.clips
    undone = editor.edit(Edit(revision=4, action="undo"))
    assert undone["clips"] == scored["clips"] and undone["match"] == scored["match"]
    redone = editor.edit(Edit(revision=5, action="redo"))
    assert redone["clips"] == updated["clips"] and redone["match"] == updated["match"]


def test_match_end_in_existing_chain_refuses_preview_and_apply(editor):
    editor.edit(Edit(revision=0, action="create", start_sec=5, end_sec=7))
    editor.edit(
        Edit(
            revision=1,
            action="set_match",
            match={
                "sides": {"A": "A", "B": "B"},
                "rules": {"type": "soft_tennis", "match_games": 7},
                "first_server": "A",
            },
        )
    )
    before = {"games": [3, 0], "points": [3, 1]}
    editor.edit(
        Edit(
            revision=2,
            action="set_score",
            name="clip_000",
            score=_anchor_winner(before=before, winner="B"),
        )
    )
    edit = Edit(
        revision=3,
        action="set_score",
        name="clip_000",
        score=_anchor_winner(before=before, winner="A"),
    )
    saved = editor.path.read_bytes()
    for apply in (editor.preview, editor.edit):
        with pytest.raises(ValueError, match="clip 'clip_001'.*試合終了後"):
            apply(edit)
        assert editor.path.read_bytes() == saved and editor.revision == 3
    editor.edit(Edit(revision=3, action="set_score", name="clip_001", score=None))
    terminal = editor.edit(edit.model_copy(update={"revision": 4}))
    assert terminal["clips"][1]["score"] is None
    assert terminal["score_views"]["clip_000"]["after"]["points_label"] == "試合終了"


@pytest.mark.parametrize("stop", ["unknown", "multiple_points", None])
def test_breaking_confirmed_chain_requires_a_new_anchor_before_export(editor, stop):
    editor.edit(Edit(revision=0, action="create", start_sec=5, end_sec=7))
    editor.edit(Edit(revision=1, action="set_match", match=_match_payload()))
    before = {"sets": [0, 0], "games": [0, 0], "points": [2, 1]}
    first = _anchor_winner(before=before, winner="A")
    editor.edit(Edit(revision=2, action="set_score", name="clip_000", score=first))
    second = {
        **first,
        "before": None,
        "before_source": "chain",
        "winner": "B",
        "note": "映像で確認",
    }
    editor.edit(Edit(revision=3, action="set_score", name="clip_001", score=second))
    linked = editor.edit(Edit(revision=4, action="create", start_sec=7, end_sec=8))
    assert linked["score_views"]["clip_001"]["after"]["points"] == [3, 2]
    assert linked["score_views"]["clip_002"]["chain_available"]
    saved = editor.path.read_bytes()

    replacement = (
        {**first, "status": stop, "winner": None, "note": "判定保留"} if stop else None
    )
    edit = Edit(revision=5, action="set_score", name="clip_000", score=replacement)
    assert editor.preview(edit) == {
        "revision": 5,
        "changed_clips": 0,
        "broken_chain_clips": ["clip_001"],
    }
    assert editor.path.read_bytes() == saved and editor.revision == 5
    broken = editor.edit(edit)
    assert broken["clips"][1]["score"] == linked["clips"][1]["score"]
    assert broken["score_views"]["clip_001"]["chain_broken"]
    assert broken["score_views"]["clip_001"]["after"] is None
    assert broken["score_views"]["clip_001"]["before_points"] is None
    assert not broken["score_views"]["clip_002"]["chain_available"]
    reloaded = _loaded(editor)
    assert reloaded.clips[1].score.before_source == "chain"
    with pytest.raises(ValueError, match="連鎖切れ.*clip_001"):
        export_scores(reloaded)
    with pytest.raises(ValueError, match="手入力"):
        editor.edit(Edit(revision=6, action="set_score", name="clip_001", score=second))

    fixed = editor.edit(
        Edit(
            revision=6,
            action="set_score",
            name="clip_001",
            score={
                **second,
                "before_source": "anchor",
                "before": {**before, "points": [3, 1]},
            },
        )
    )
    assert not fixed["score_views"]["clip_001"]["chain_broken"]
    assert fixed["score_views"]["clip_001"]["after"]["points"] == [3, 2]
    assert fixed["clips"][1]["score"]["note"] == "映像で確認"
    assert fixed["clips"][2]["score"]["before"]["points"] == [3, 2]
    assert "clip_001,confirmed," in export_scores(_loaded(editor))
    assert editor.edit(Edit(revision=7, action="undo"))["score_views"]["clip_001"][
        "chain_broken"
    ]
    restored = editor.edit(Edit(revision=8, action="undo"))
    assert restored["score_views"]["clip_001"] == linked["score_views"]["clip_001"]
    assert "clip_001,confirmed," in export_scores(_loaded(editor))


def test_csv_import_undo_and_rename_move_delete_keep_clip_identity(editor):
    editor.edit(Edit(revision=0, action="create", start_sec=5, end_sec=7))
    editor.edit(Edit(revision=1, action="set_match", match=_match_payload()))
    csv_text = export_scores(editor.project) + (
        "clip_000,confirmed,0,0,0,0,1,0,A,A,first\r\n"
        "clip_001,confirmed,0,0,0,0,2,0,A,B,second\r\n"
    )
    imported = editor.edit(Edit(revision=2, action="import_scores", csv_text=csv_text))
    assert editor.edit(Edit(revision=3, action="undo"))["clips"][0]["score"] is None
    assert all(clip.score is None for clip in editor.project.clips)
    assert editor.edit(Edit(revision=4, action="redo"))["clips"] == imported["clips"]
    renamed = editor.edit(
        Edit(
            revision=5,
            action="update",
            name="clip_000",
            new_name="moved",
            start_sec=7,
            end_sec=9,
        )
    )
    assert renamed["clips"][0]["score"] == imported["clips"][0]["score"]
    assert renamed["clips"][1] == imported["clips"][1]
    assert [clip.score.note for clip in _loaded(editor).clips] == ["first", "second"]
    editor.edit(Edit(revision=6, action="delete", name="moved"))
    assert editor.snapshot()["clips"] == [imported["clips"][1]]


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
