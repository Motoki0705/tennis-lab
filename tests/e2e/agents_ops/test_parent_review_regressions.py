"""Known evidence cannot disappear, and completion must reach the parent."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_human_agreement import delegation_args
from tests.e2e.agents_ops.test_merge_gate import GateEnv, make_gate_env
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_notifications import notification, notify
from tests.e2e.agents_ops.test_parent_self_work import completion_spec
from tests.e2e.agents_ops.test_parent_session import (
    OPS,
    ParentEnv,
    evidence_comments,
    evidence_number,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_review_recovery import next_request


def interrupted_notify(p: ParentEnv, spec: dict) -> None:
    wrapper = p.root / "interrupt-notification.py"
    wrapper.write_text(
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(OPS)!r})\n"
        "import parent.record_writer as writer\n"
        "original = writer.post_once\n"
        "def interrupt(session, issue, body, key, **kwargs):\n"
        "    if issue == session.parent and key.startswith('notification:'):\n"
        "        raise ValueError('interrupted before parent POST intent')\n"
        "    return original(session, issue, body, key, **kwargs)\n"
        "writer.post_once = interrupt\n"
        "runpy.run_module('parent', run_name='__main__')\n"
    )
    path = p.root / "interrupted-spec.json"
    path.write_text(json.dumps(spec))
    result = subprocess.run(
        [
            sys.executable,
            str(wrapper),
            "--repo",
            "o/r",
            "--parent",
            "1",
            "notify",
            "--spec",
            str(path),
        ],
        env=p.env,
        cwd=p.repo,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and "interrupted before parent POST" in result.stderr
    assert not p.state()["comments"]["1"]


@pytest.mark.parametrize("fault", ["normal", "deleted", "changed"])
def test_mixed_history_revocation_integrity_guards_delegate_review_and_merge(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    p = parent_env
    agreed = p.agreed

    def legacy_agreement() -> None:
        agreed()
        state = p.state()
        dest = str(evidence_number(state))
        # Pre-journal parent evidence; later review/revocation receipts remain intact.
        state["comments"]["1"] = state["comments"][dest]
        state["comments"][dest] = []
        p.save(state)
        directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
        shutil.rmtree(directory / "comment-outbox")

    monkeypatch.setattr(p, "agreed", legacy_agreement)
    g = make_gate_env(p, monkeypatch, "multiple")
    assert all(
        r["decision"] == "merge"
        for r in g.results(g.cli("merge", AGENT_AUTO_MERGE="0")).values()
    )
    p.child("independent")
    review = next_request(g.review)
    tree = p.worktree("independent")
    if fault != "normal":
        p.cli(
            "basis-revoke",
            "--reason-file",
            p.note("人間との再合意まで全体を停止する。"),
        )
        state = p.state()
        revoked = next(
            c
            for c in evidence_comments(state)
            if '"event": "basis_revoked"' in c["body"]
        )
        if fault == "deleted":
            evidence_comments(state).remove(revoked)
        else:
            revoked["body"] += "\nremote edit"
        p.save(state)
        denied = p.cli(*delegation_args(tree, child=13), ok=False)
        rejected_review = review.run(ok=False)
        rejected_merge = g.cli("merge", AGENT_AUTO_MERGE="0")
        for result in (denied, rejected_review, rejected_merge):
            assert result.returncode == 2 and "comment journal" in result.stderr
        assert len(g.review.calls()) == 1 and not g.merge_calls()
        return
    assert (
        json.loads(p.cli(*delegation_args(tree, child=13)).stdout)["action"] == "launch"
    )
    assert json.loads(review.run().stdout)["publication"] == "published"
    current = replace(g, review=review)
    current.confirm()
    assert all(
        r["decision"] == "merge"
        for r in current.results(current.cli("merge", AGENT_AUTO_MERGE="0")).values()
    )


@pytest.mark.parametrize("gate_env", ["standalone"], indirect=True)
def test_completion_waits_for_parent_publication_then_recovers_once(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    p.cli(
        "record-result",
        "--child",
        "1",
        "--evidence-file",
        p.note("親が受入条件と差分・テストを確認した。"),
    )
    assert g.cli("merge").returncode == 0
    spec = notification(p, "completion", "completion")
    spec["details"]["achievements"] = [
        "Standalone implementation, real merge, and parent acceptance"
    ]
    spec["details"]["remaining"] = []
    interrupted_notify(p, spec)
    status = json.loads(p.cli("status").stdout)
    assert status["assignment"] == "active"
    assert status["notifications"][0]["state"] == "pending"
    report = completion_spec(p, accepted=True)
    refused = p.cli("workflow-report", "--spec", report, ok=False)
    ended = p.cli(
        "release", "--ended", "--note-file", p.note("全体受入後に終了する。"), ok=False
    )
    assert "not published" in refused.stderr and "not published" in ended.stderr
    assert json.loads(p.cli("status").stdout)["assignment"] == "active"
    notify(p, spec, extra={"PARENT_FAIL_AFTER_COMMENT": "1"}, ok=False)
    notify(p, spec)
    assert len(p.state()["comments"]["1"]) == 1
    assert (
        json.loads(p.cli("status").stdout)["notifications"][0]["state"] == "published"
    )
    assert (
        json.loads(p.cli("workflow-report", "--spec", report).stdout)["state"]
        == "completed"
    )
    p.cli("release", "--ended", "--note-file", p.note("全体受入後に終了する。"))
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"
    assert len(p.state()["comments"]["1"]) == 1


def test_status_exposes_an_unpublished_question_without_mutating_it(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    spec = notification(p, "question", "question-pending")
    interrupted_notify(p, spec)
    original = p.state()
    entry = json.loads(p.cli("status").stdout)["notifications"][0]
    assert (
        entry["key"] == "question-pending"
        and entry["state"] == "pending"
        and entry["url"] is None
    )
    assert entry["record_url"].startswith(
        f"https://github.com/o/r/issues/{evidence_number(p.state())}#issuecomment-"
    )
    assert p.state() == original
    notify(p, spec)
    assert (
        json.loads(p.cli("status").stdout)["notifications"][0]["state"] == "published"
    )
    assert len(p.state()["comments"]["1"]) == 1
