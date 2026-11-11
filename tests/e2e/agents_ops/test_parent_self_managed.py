"""Autonomous self-work uses the same formal target and worker session."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_dispatcher_supervisor import DispatchHarness
from tests.e2e.agents_ops.test_dispatcher_supervisor import d as d
from tests.e2e.agents_ops.test_parent_development import (
    child_tree,
    drain,
    launch,
    workflow,
)


def test_labelled_self_worker_continues_and_split_cancels_new_turn(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use the real dispatcher judgement route for a formal parent with zero
    # children, instead of relabelling an admitted standalone assignment.
    monkeypatch.setattr(d, "prepare", lambda: d.prepare_family())
    mod, session, env = workflow(d, monkeypatch)
    tree = child_tree(d)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=1,
        prompt="同じIssueの成果を実装・検証する。",
        model=None,
        branch="codex/parser",
        worktree=tree,
    )
    try:
        launch(d, monkeypatch)
        first_job = drain(d, first["job_id"])
        previous = json.loads(
            (Path(first_job["directory"]) / "runtime/result.json").read_text()
        )
        followup = mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=1,
            prompt="同じ成果の追加確認を行う。",
            model=None,
            continuation="前ターンの終了・回収を確認した。",
        )
        launch(d, monkeypatch)
        next_job = drain(d, followup["job_id"])
        current = json.loads(
            (Path(next_job["directory"]) / "runtime/result.json").read_text()
        )
        assert current["runtime"]["session_id"] == previous["runtime"]["session_id"]
        assert next_job["resume_from"] == str(Path(first_job["directory"]) / "runtime")
        assert d.row()["dispatch"]["candidate"]["route"] == "parent"
        assert d.h.gh.parents == {}
        assert {label["name"] for label in d.h.gh.issues[1]["labels"]} == {
            "issue:parent"
        }
        assert all(
            not (
                write["method"] == "POST"
                and (
                    write["endpoint"].endswith("/sub_issues")
                    or write["endpoint"] == "repos/o/r/issues"
                )
            )
            for write in d.h.gh.writes
        )
        reserved = mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=1,
            prompt="起動直前に正式関係を再確認する。",
            model=None,
            continuation="同じworkerへ追加確認を依頼。",
        )
        # An external split after reservation cannot start either self-work or a
        # replacement worker for the new child using the saved self reservation.
        child = copy.deepcopy(d.h.gh.issues[1])
        child.update(
            number=10,
            id=10010,
            title="[開発 #1] 新しい成果",
            labels=[{"name": "issue:child"}],
        )
        d.h.gh.issues[10] = child
        d.h.gh.parents[10] = 1
        launch(d, monkeypatch)
        cancelled = drain(d, reserved["job_id"])
        assert cancelled["cancelled_before_start"]
        assert not (Path(cancelled["directory"]) / "runtime").exists()
        assert (
            d.row()["tasks"]["1"]["worker_ended"]["delegation_id"]
            == cancelled["delegation_id"]
        )
        with (
            session.store.locked(),
            pytest.raises(ValueError, match="work targets changed"),
        ):
            session.work_items()
    finally:
        d.control.reconcile(stop=True)
