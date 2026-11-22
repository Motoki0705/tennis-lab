"""Judgement publication through existing delegation, review and merge gates."""

from __future__ import annotations

import importlib
import json

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import decision
from tests.e2e.agents_ops.test_execution_basis import (
    ignored_comment,
    record,
    registry,
    spec,
)
from tests.e2e.agents_ops.test_human_agreement import agreements, delegation_args
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    evidence_comments,
    untracked_legacy_fixture,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def publish(
    p: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    disposition: str,
    rationale: str = "既存の管理先で作業を進める。",
) -> str:
    judgement = importlib.import_module("dispatcher.judgement")
    collector = importlib.import_module("dispatcher.collect")
    sessions = importlib.import_module("parent.session")
    state = p.state()
    d = decision(n, disposition, target=state["parents"].get(str(n)))
    d["rationale"] = rationale
    # This test exercises publication itself; source identities are the same ones
    # the collector hands to publication, backed by the fixture's fake GitHub.
    code = collector.code_snapshot(p.repo)
    d["sources"] = [
        f"code:{code['commit']}:input.txt" if s.startswith("code:") else s
        for s in d["sources"]
    ]
    snapshot = {"repo": "o/r", "issues": state["issues"], "code": code}
    with monkeypatch.context() as env:
        for key, value in p.env.items():
            env.setenv(key, value)
        store = sessions.Store(registry(p).parent)
        with store.locked():
            session = sessions.ParentSession(sessions.GitHub("o/r"), store, n, None)
            return str(judgement.publish_decision(session, d, snapshot))


@pytest.mark.parametrize("legacy_ignored", [False, True])
def test_judgement_update_preserves_autonomous_delegation(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    legacy_ignored: bool,
) -> None:
    p = parent_env
    assert publish(p, monkeypatch, 1, "startable") == "created"
    display = p.comments()[0]
    ignored = [ignored_comment(display)] if legacy_ignored else []
    value = spec(p, ignored)
    recorded = record(p, value)  # Record before assignment, as the dispatcher does.
    p.cli("register")
    p.child("worker")
    frozen = registry(p).read_bytes()
    assert publish(p, monkeypatch, 1, "existing_owner") == "updated"
    assert registry(p).read_bytes() == frozen
    launch = json.loads(p.cli(*delegation_args(p.worktree("worker"))).stdout)
    assert launch["action"] == "launch"
    assert (
        launch["assignment"]["authorization"]["execution_basis_sha256"]
        == (recorded["basis_sha256"])
    )
    assert launch["assignment"]["authorization"]["generation"] == recorded["generation"]
    policy = importlib.import_module("parent.policy")
    basis_events = [
        e
        for e in policy.events(p.comments(), "operator", 1)
        if e["event"] == "execution_basis"
    ]
    assert len(basis_events) == 1
    assert basis_events[0]["ignored_comments"] == ignored


def test_legacy_human_snapshot_with_judgement_remains_executable(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    publish(p, monkeypatch, 1, "information_needed")
    display = p.comments()[0]
    p.agreed()
    policy = importlib.import_module("parent.policy")
    execution = importlib.import_module("parent.execution")
    event = agreements(p)[-1]
    body = event.pop("body")
    # Old human snapshots included the current judgement as a normal comment.
    event["comment_inputs"].append(
        {
            "id": display["id"],
            "login": "operator",
            "sha256": policy.digest(execution.comment_snapshot(display)),
        }
    )
    state = p.state()
    next(c for c in evidence_comments(state) if c["body"] == body)["body"] = (
        policy.marker(event)
    )
    p.save(state)
    untracked_legacy_fixture(p)
    publish(p, monkeypatch, 1, "human_hold")
    p.child("legacy")
    launch = json.loads(p.cli(*delegation_args(p.worktree("legacy"))).stdout)
    assert launch["action"] == "launch"
