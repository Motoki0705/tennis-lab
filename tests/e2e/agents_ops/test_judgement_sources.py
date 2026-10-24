"""Cross-subject citations remain fresh through publication and reservation."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import Harness, decision, issue
from tests.e2e.agents_ops.test_dispatcher_judgement import h as h

CHANGES = [
    ("issue", "edited"),
    ("comment", "edited"),
    ("comment", "deleted"),
    ("system_comment", "edited"),
    ("system_comment", "deleted"),
    ("pr", "edited"),
    ("pr", "deleted"),
    ("pr", "retargeted"),
    ("source_dependency", "edited"),
]


def note_body(text: str) -> str:
    policy = importlib.import_module("parent.policy")
    return str(
        policy.marker(
            {
                "parent": 2,
                "event": "note",
                "generation": 1,
                "body_sha256": policy.digest(text),
            }
        )
        + "\n\n"
        + text
    )


def cite(h: Harness, kind: str) -> dict[str, Any]:
    h.gh.issues.update(
        {2: issue(2), 3: issue(3), 200: issue(200, state="closed", reason="completed")}
    )
    h.gh.dependencies[2] = [200]
    h.gh.comment_rows[2] = [
        {
            "id": 700,
            "user": {"login": "operator"},
            "body": note_body("確認済みの進捗。")
            if kind == "system_comment"
            else "関連する指示。",
        }
    ]
    first: dict[str, Any] = decision(1)
    if kind == "pr":
        h.gh.prs = [
            {
                "number": 300,
                "body": "Closes #2",
                "state": "open",
                "merged_at": None,
                "head": {"ref": "codex/other-work", "sha": "a" * 40},
            }
        ]
        reference = "pr:300"
    elif kind in {"comment", "system_comment"}:
        reference = "comment:2:700"
    else:
        reference = "issue:2"
    first["sources"].append(reference)
    h.response(
        first,
        decision(
            2, "existing_owner" if kind == "pr" else "information_needed", deps=(200,)
        ),
        decision(3),
    )
    return first


def change_source(h: Harness, kind: str, change: str) -> None:
    if kind == "issue":
        h.gh.issues[2]["body"] += "\n新しい要件。"
    elif kind == "source_dependency":
        h.gh.issues[200]["state_reason"] = "not_planned"
    elif kind in {"comment", "system_comment"}:
        if change == "deleted":
            h.gh.comment_rows[2].clear()
        else:
            h.gh.comment_rows[2][0]["body"] = (
                note_body("変更後の進捗。")
                if kind == "system_comment"
                else "指示を変更した。"
            )
    elif change == "deleted":
        h.gh.prs.clear()
    elif change == "retargeted":
        h.gh.prs[0]["body"] = "Closes #200"
    else:
        h.gh.prs[0]["head"]["sha"] = "b" * 40


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize(("kind", "change"), CHANGES)
def test_cited_subject_changes_defer_readers_before_publication(
    h: Harness, apply: bool, kind: str, change: str
) -> None:
    cite(h, kind)

    def concurrent_edit(endpoint: str) -> None:
        if h.calls.exists():
            h.gh.on_get = None
            change_source(h, kind, change)

    h.gh.on_get = concurrent_edit
    result = h.run(apply=apply, runner=h.fake_runner)
    assert result["result"] == ("recorded" if apply else "preview"), result
    # Only an explicit citation makes another subject's progress marker material.
    deferred = [1] if kind == "system_comment" else [1, 2]
    assert [(d["issue"], d["stage"]) for d in result["deferred"]] == [
        (n, "preflight") for n in deferred
    ]
    assert [d["issue"] for d in result["decisions"]] == [
        n for n in (1, 2, 3) if n not in deferred
    ]
    assert [c["parent"] for c in result["candidates"]] == ([3] if apply else [])
    assert 1 not in h.gh.comment_rows
    assert len(h.calls.read_text().splitlines()) == 1


@pytest.mark.parametrize(("kind", "change"), CHANGES)
def test_published_candidate_rechecks_cited_sources_before_reservation(
    h: Harness, kind: str, change: str
) -> None:
    first = cite(h, kind)
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    candidates = {c["parent"]: c for c in result["candidates"]}
    assert set(candidates) == {1, 3}
    candidate, snapshot = candidates[1], h.snapshot()
    assert candidate["target_sha256"] == h.collector.digest(
        h.collector.material(snapshot, 1, first["sources"])
    )
    assert (
        candidate["target_sha256"]
        != snapshot["execution_inputs"]["1"]["context"]["dispatcher"]["target_sha256"]
    )
    with h.store.locked():
        h.gh.calls.clear()
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)
        if kind != "pr":
            assert "repos/o/r/issues/2" in h.gh.calls
            assert "repos/o/r/issues/200" in h.gh.calls
        assert any("repos/o/r/pulls" in endpoint for endpoint in h.gh.calls)
        execution = importlib.import_module("parent.execution")
        session_class = importlib.import_module("parent.session").ParentSession
        auth = execution.validate_candidate(
            session_class(h.gh, h.store, 1, None),
            candidate["basis_sha256"],
            candidate["generation"],
        )
        assert auth["execution_basis_sha256"] == candidate["basis_sha256"]
    change_source(h, kind, change)
    with h.store.locked(), pytest.raises(h.collector.InputsChanged):
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)
    with h.store.locked():
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidates[3])
    assert len(h.calls.read_text().splitlines()) == 1


@pytest.mark.parametrize("stage", ["publication", "final_check"])
def test_cited_source_changes_during_writes_remove_the_reader_candidate(
    h: Harness, stage: str
) -> None:
    cite(h, "comment")

    def concurrent_edit() -> None:
        if stage == "final_check" and not any(
            "execution_basis" in w["payload"]["body"].splitlines()[0]
            for w in h.gh.writes
        ):
            return
        h.gh.on_write = None
        change_source(h, "comment", "deleted")

    h.gh.on_write = concurrent_edit
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    assert [c["parent"] for c in result["candidates"]] == [3]
    assert [d["issue"] for d in result["decisions"]] == [3]
    assert next(d for d in result["deferred"] if d["issue"] == 1)["stage"] == stage
    assert {p["issue"] for p in result["publications"]} == (
        {1, 3} if stage == "publication" else {1, 2, 3}
    )
    assert any(
        "execution_basis" in c["body"].splitlines()[0] for c in h.gh.comment_rows[1]
    ) == (stage == "final_check")


def test_missing_cited_issue_cannot_reuse_a_published_candidate(h: Harness) -> None:
    cite(h, "issue")
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    candidate, snapshot = result["candidates"][0], h.snapshot()
    del h.gh.issues[2]
    h.gh.fail = "repos/o/r/issues/2"  # Fake the real endpoint's deletion/404.
    with h.store.locked(), pytest.raises(ValueError, match="API failure"):
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)


def test_reference_forms_have_identical_target_material(h: Harness) -> None:
    d = cite(h, "comment")
    result = h.run(runner=h.fake_runner)
    assert result["result"] == "recorded", result
    snapshot, candidate = h.snapshot(), result["candidates"][0]
    assert h.collector.material(snapshot, 1, d["sources"]) == h.collector.material(
        snapshot, 1, candidate["sources"]
    )
