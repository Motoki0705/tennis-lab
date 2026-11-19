"""One family observation per launch plan, including its fresh admission check."""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv, evidence_number
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def test_register_returns_assignment_without_unrelated_status_traversal(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    data = json.loads(p.cli("register").stdout)
    assert data["assignment"] == "active"
    assert data["phase"] == "担当指定のみ"
    assert data["current_session_is_owner"] is True
    assert data["parent_action_applied"] is True
    assert not any(
        "/sub_issues" in arg or "/timeline" in arg for call in p.calls() for arg in call
    )
    assert len([c for c in p.comments() if '"event": "assigned"' in c["body"]]) == 1


def test_status_keeps_distinct_binding_family_and_pr_observations(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("implementation")
    offset = len(p.calls())
    assert json.loads(p.cli("status").stdout)["launch_plan"]["tasks"][0]["child"] == 10
    endpoints = [
        a
        for c in p.calls()[offset:]
        for a in c
        if a == "user" or a.startswith("repos/")
    ]
    # The cold CLI needs one identity, five family/relation reads, four history
    # reads (including verification of the saved binding), and two timelines.
    # These 12 distinct facts cannot be removed as duplicate observations.
    assert len(endpoints) == len(set(endpoints)) == 12
    assert f"repos/o/r/issues/{evidence_number(p.state())}" in endpoints


@pytest.mark.parametrize("change", ["proposal-label", "evidence-marker"])
def test_new_record_still_checks_that_its_parent_is_a_work_issue(
    parent_env: ParentEnv,
    change: str,
) -> None:
    p = parent_env
    p.cli("register")
    state = p.state()
    parent = state["issues"]["1"]
    if change == "proposal-label":
        parent["labels"].append({"name": "issue:proposal"})
    else:
        parent["body"] = (
            '<!-- parent-evidence:v1 {"repo": "o/r", "parent": 2} -->\n'
            + parent["body"]
        )
    p.save(state)
    offset = len(p.calls())
    failed = p.cli(
        "note", "--body-file", p.note("作業Issue以外へ投稿しない。"), ok=False
    )
    assert "cannot be used as a work issue" in failed.stderr
    assert not any("POST" in c for c in p.calls()[offset:])


@pytest.mark.parametrize("uncertain", [False, True])
def test_registration_binds_existing_containers_without_recreating_them(
    parent_env: ParentEnv,
    uncertain: bool,
) -> None:
    p = parent_env
    created = p.cli(
        "evidence",
        "--create",
        extra={"PARENT_FAIL_AFTER_CREATE": "1"} if uncertain else {},
        ok=not uncertain,
    )
    if uncertain:
        assert "uncertain" in created.stderr
    directory = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/evidence-issues"
    )
    journals = list(directory.glob("*.json"))
    assert len(journals) == 1
    if uncertain:
        assert json.loads(journals[0].read_text())["state"] == "sent"
    else:
        # Model a container created on another host, without a local binding.
        journals[0].unlink()
    assert json.loads(p.cli("register").stdout)["assignment"] == "active"
    assert json.loads(journals[0].read_text())["state"] == "bound"
    assert len([c for c in p.calls() if "POST" in c and "repos/o/r/issues" in c]) == 1


def test_delegation_shares_inputs_within_its_independent_fresh_check(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = parent_env
    p.agreed()
    p.child("implementation")
    worktree = p.worktree("cost")
    monkeypatch.chdir(p.repo)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    store = importlib.import_module("parent.state").Store(
        importlib.import_module("parent.state").state_directory()
    )
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    snapshot = importlib.import_module("parent.snapshot")
    validate_worktree = snapshot.validate_worktree
    admission = []

    def validate(*args: Any) -> None:
        validate_worktree(*args)
        admission.append(len(p.calls()))

    monkeypatch.setattr(snapshot, "validate_worktree", validate)
    try:
        with store.locked(), gh.history_read_scope():
            session = importlib.import_module("parent.session").ParentSession(
                gh, store, 1, "claude:private-session-a-unique"
            )
            session.delegate(10, "codex", "codex/cost", worktree, None, None)
    finally:
        gh.close()
    calls = p.calls()[admission[0] :]
    write = next(
        i for i, c in enumerate(calls) if "PATCH" in c and "repos/o/r/issues/10" in c
    )
    endpoints = [a for c in calls[:write] for a in c if a.startswith("repos/")]
    assert endpoints.count("repos/o/r/issues/1") == 1
    # The child is read once for admission and independently once by update_body
    # immediately before its PATCH. This second freshness check must remain.
    assert endpoints.count("repos/o/r/issues/10") == 2
    assert endpoints.count("repos/o/r/issues/1/comments?per_page=100&page=1") == 1
    evidence = evidence_number(p.state())
    assert (
        endpoints.count(f"repos/o/r/issues/{evidence}/comments?per_page=100&page=1")
        == 1
    )
    assert endpoints.count("repos/o/r/issues/1/sub_issues?per_page=100&page=1") == 1


def test_record_checks_the_fresh_rows_after_acquiring_its_outbox_lock(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = parent_env
    p.cli("register")
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    writer = importlib.import_module("parent.record_writer")
    original = writer.locked
    evidence = evidence_number(p.state())

    @contextmanager
    def locked(path: Path) -> Any:
        with original(path):
            if path.parent.name == "comment-outbox":
                state = p.state()
                state["comments"][str(evidence)].append(
                    {
                        "id": 99999,
                        "body": "<!-- parent-session:v2 {} -->",
                        "user": {"login": "operator"},
                    }
                )
                p.save(state)
            yield

    monkeypatch.setattr(writer, "locked", locked)
    store = importlib.import_module("parent.state").Store(
        importlib.import_module("parent.state").state_directory()
    )
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    offset = len(p.calls())
    try:
        with store.locked(), gh.history_read_scope():
            session = importlib.import_module("parent.session").ParentSession(
                gh, store, 1, "claude:private-session-a-unique"
            )
            with pytest.raises(ValueError, match="unknown|malformed"):
                session.note("検証前に投稿しない。")
    finally:
        gh.close()
    assert not any("POST" in c for c in p.calls()[offset:])


def test_fresh_launch_plan_observes_each_family_relation_once(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    p.agreed()
    p.child("implementation")
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    directory = importlib.import_module("parent.state").state_directory()
    store = importlib.import_module("parent.state").Store(directory)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    try:
        with store.locked():
            session = importlib.import_module("parent.session").ParentSession(
                gh, store, 1, "claude:private-session-a-unique"
            )
            for _ in range(2):
                offset = len(p.calls())
                with gh.fresh_reads():
                    assert session.launch_plan()["tasks"][0]["child"] == 10
                calls = p.calls()[offset:]
                endpoints = [a for c in calls for a in c if a.startswith("repos/")]
                for n in (1, 10):
                    assert (
                        endpoints.count(
                            f"repos/o/r/issues/{n}/sub_issues?per_page=100&page=1"
                        )
                        == 1
                    )
                    assert endpoints.count(f"repos/o/r/issues/{n}/parent") == 1
                    assert (
                        endpoints.count(
                            f"repos/o/r/issues/{n}/timeline?per_page=100&page=1"
                        )
                        == 1
                    )
    finally:
        gh.close()
