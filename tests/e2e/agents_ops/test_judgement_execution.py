"""Judgement publication through existing delegation, review and merge gates."""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import decision
from tests.e2e.agents_ops.test_execution_basis import (
    ignored_comment,
    ordinary_comment,
    record,
    registry,
    spec,
)
from tests.e2e.agents_ops.test_human_agreement import agreements, delegation_args
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv
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


@pytest.mark.parametrize("gate_env", ["multiple"], indirect=True)
def test_judgement_publication_preserves_human_delegate_review_and_merge(
    gate_env: GateEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    g, p = gate_env, gate_env.review.p
    frozen = (g.review.run_dir / "result.json").read_bytes(), registry(p).read_bytes()
    for n, disposition in ((1, "human_hold"), (10, "existing_parent")):
        assert publish(p, monkeypatch, n, disposition) == "created"
        assert (
            publish(p, monkeypatch, n, disposition, "判定理由の表示を更新した。")
            == "updated"
        )
    assert (
        (g.review.run_dir / "result.json").read_bytes(),
        registry(p).read_bytes(),
    ) == frozen
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert all(row["decision"] == "merge" for row in rows.values())
    assert g.merge_calls() == [] and len(g.review.calls()) == 1
    # Main's delegate is the reservation path also used by parallel child launch.
    p.child("next")
    launch = json.loads(p.cli(*delegation_args(p.worktree("next"), child=13)).stdout)
    assert launch["action"] == "launch"
    assert launch["assignment"]["authorization"] == g.review.request["authorization"]
    prepared = json.loads(
        p.cli(
            "review-prepare",
            "--prs",
            "100",
            "101",
            "102",
            "--reason-file",
            p.note("判定表示の更新後もレビュー準備できることをfakeで検査する。"),
        ).stdout
    )
    assert prepared["authorization"] == g.review.request["authorization"]


@pytest.mark.parametrize("legacy_ignored", [False, True])
def test_judgement_update_preserves_autonomous_delegation(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    legacy_ignored: bool,
) -> None:
    p = parent_env
    assert publish(p, monkeypatch, 1, "startable") == "created"
    display = p.state()["comments"]["1"][0]
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
        for e in policy.events(p.state()["comments"]["1"], "operator", 1)
        if e["event"] == "execution_basis"
    ]
    assert len(basis_events) == 1
    assert basis_events[0]["ignored_comments"] == ignored


def test_legacy_human_snapshot_with_judgement_remains_executable(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    publish(p, monkeypatch, 1, "information_needed")
    display = p.state()["comments"]["1"][0]
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
    next(c for c in state["comments"]["1"] if c["body"] == body)["body"] = (
        policy.marker(event)
    )
    p.save(state)
    publish(p, monkeypatch, 1, "human_hold")
    p.child("legacy")
    launch = json.loads(p.cli(*delegation_args(p.worktree("legacy"))).stdout)
    assert launch["action"] == "launch"


@pytest.mark.parametrize(
    "change",
    [
        "plain",
        "quoted",
        "prefix_only",
        "appended",
        "digest",
        "publisher",
        "issue",
        "repo",
    ],
)
@pytest.mark.parametrize("gate_env", ["multiple"], indirect=True)
def test_forged_judgement_and_human_instructions_still_stop_work(
    gate_env: GateEnv, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    publish(p, monkeypatch, 1, "human_hold")
    state = p.state()
    display = state["comments"]["1"][-1]
    body = display["body"]
    login = "operator"
    if change == "plain":
        body = "この作業を停止してください。"
    elif change == "quoted":
        body = "人間の指示: 作業停止。\n\n" + body
    elif change == "prefix_only":
        body = "<!-- issue-start-judgement:v1 作業停止 -->\n\n作業を止める。"
    elif change == "appended":
        body += "\n\n追加指示: 作業を停止してください。"
    elif change == "publisher":
        login = "another-publisher"
    else:
        records = importlib.import_module("dispatcher.records")
        data: dict[str, Any] = json.loads(
            body.splitlines()[0][len(records.PREFIX) : -4]
        )
        if change == "digest":
            data["detail_sha256"] = "0" * 64
            body = records.PREFIX + json.dumps(data) + " -->" + body[body.index("\n") :]
        else:
            body = records.comment_body(
                "other/repository" if change == "repo" else "o/r",
                2 if change == "issue" else 1,
                data["detail"],
            )
    ordinary_comment(p, body, login)
    before = p.state(), registry(p).read_bytes()
    responses = [
        p.cli(*delegation_args(p.worktree("blocked")), ok=False),
        p.cli("review-prepare", "--prs", "100", "101", "102", ok=False),
        g.cli("merge"),
    ]
    assert all(response.returncode == 2 for response in responses)
    assert (p.state(), registry(p).read_bytes()) == before
    assert g.merge_calls() == [] and len(g.review.calls()) == 1
