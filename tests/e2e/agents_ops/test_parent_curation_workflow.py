"""Parent timeline outcomes across normal review and an in-flight legacy run."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv, ci_rollup
from tests.e2e.agents_ops.test_parent_session import (
    ReviewEnv,
    evidence_comments,
    evidence_number,
    review_policy_module,
)
from tests.e2e.agents_ops.test_parent_session import pair_review_env as pair_review_env
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)


def ready_gate(r: ReviewEnv, monkeypatch: pytest.MonkeyPatch) -> GateEnv:
    state = r.p.state()
    for pr in state["prs"]:
        pr.update(labels=[], mergeable=True, statusCheckRollup=ci_rollup("success"))
    r.p.save(state)
    return GateEnv(r, review_policy_module(monkeypatch))


def test_zero_finding_review_five_operations_leave_parent_timeline_empty(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, p = single_review_env, single_review_env.p
    assert not p.state()["comments"]["1"]
    result = json.loads(r.run().stdout)
    assert result["publication"] == "published"
    run_id = r.request["review_run_id"]
    reason = p.note("結果を受領し、指摘ゼロ・対象HEADのCIと差分を親が確認した。")
    for state in ("result_received", "fixes_verified"):
        p.cli(
            "review-state",
            "--run-id",
            run_id,
            "--state",
            state,
            "--evidence-file",
            reason,
        )
    policy = review_policy_module(monkeypatch)
    session_policy = importlib.import_module("parent.policy")
    events = session_policy.events(evidence_comments(p.state()), "operator", 1)
    assert [e["state"] for e in events if e.get("event") == "review"] == [
        "targets_fixed",
        "instructed",
        "result_received",
        "fixes_verified",
    ]
    reviews = policy.records(r.comments(1), "operator")
    assert len(reviews) == 1 and reviews[0]["kind"] == "result"
    assert len(policy.records(r.comments(100), "operator")) == 1
    gate = ready_gate(r, monkeypatch)
    assert (
        gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"]
        == "merge"
    )
    before = copy.deepcopy(p.state()["comments"])
    r.cli("resume-publication", "--run", run_id)
    p.cli(
        "review-state",
        "--run-id",
        run_id,
        "--state",
        "fixes_verified",
        "--evidence-file",
        reason,
    )
    assert p.state()["comments"] == before and len(r.calls()) == 1
    assert not p.state()["comments"]["1"]
    assert f"/issues/{evidence_number(p.state())}" in p.state()["issues"]["1"]["body"]


def test_inflight_legacy_publication_keeps_its_permalink_then_new_records_use_evidence(
    pair_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, p = pair_review_env, pair_review_env.p
    response = json.loads(
        r.run(
            ok=False, extra={"PARENT_FAIL_PATH": "repos/o/r/issues/101/comments"}
        ).stdout
    )
    assert response["publication"] == "publication_failed"
    immutable = (r.run_dir / "result.json").read_bytes()
    policy = review_policy_module(monkeypatch)
    state = p.state()
    container = evidence_number(state)
    rows = state["comments"].pop(str(container))
    pending = next(c for c in rows if policy.records([c], "operator"))
    original_url = (
        f"https://github.com/o/r/issues/{container}#issuecomment-{pending['id']}"
    )
    legacy_url = f"https://github.com/o/r/issues/1#issuecomment-{pending['id']}"
    # Construct a pre-upgrade partial publication: legacy parent records and one
    # existing PR projection. No cross-medium copy or new-runtime outbox existed.
    state["comments"]["1"] = rows
    del state["issues"][str(container)]
    for comment in state["comments"].get("100", []):
        comment["body"] = comment["body"].replace(original_url, legacy_url)
    state["issues"]["1"]["body"] = importlib.import_module(
        "parent.policy"
    ).replace_section(
        state["issues"]["1"]["body"], "進捗概要", "Legacy publication in progress."
    )
    p.save(state)
    execution_path = r.run_dir / "execution.json"
    execution_path.write_text(
        execution_path.read_text().replace(original_url, legacy_url)
    )
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    for name in ("evidence-issues", "comment-outbox"):
        shutil.rmtree(directory / name)
    original = {c["id"]: c["body"] for c in rows}
    before_posts = [
        c for c in p.calls() if "POST" in c and "repos/o/r/issues/1/comments" in c
    ]
    resumed = json.loads(
        r.cli("resume-publication", "--run", r.request["review_run_id"]).stdout
    )
    assert resumed["publication"] == "published" and resumed["agent_invocations"] == 0
    assert (r.run_dir / "result.json").read_bytes() == immutable and len(r.calls()) == 1
    after = {c["id"]: c["body"] for c in p.state()["comments"]["1"]}
    assert after.keys() == original.keys()
    assert all(
        after[cid] == body for cid, body in original.items() if cid != pending["id"]
    )
    committed = next(c for c in p.state()["comments"]["1"] if c["id"] == pending["id"])
    assert policy.records([committed], "operator")[0]["kind"] == "result"
    assert json.loads(execution_path.read_text())["parent_run_url"] == legacy_url
    gate = ready_gate(r, monkeypatch)
    gate.confirm()  # First new post after upgrade uses the evidence writer.
    assert evidence_comments(p.state()) and len(p.state()["comments"]["1"]) == len(rows)
    assert all(
        v["decision"] == "merge"
        for v in gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0")).values()
    )
    assert [
        c for c in p.calls() if "POST" in c and "repos/o/r/issues/1/comments" in c
    ] == before_posts
