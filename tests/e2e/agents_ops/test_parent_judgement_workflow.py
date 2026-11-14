"""Parent judgement, independent lifecycle facts and resumable operations.

Reuse the fake GitHub/runner and throwaway git fixture. No live agent or network.
These scenarios distinguish content observations from authority and execution.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import (
    ended,
    latest,
    manual,
    run,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import (
    reserved as reserved,
)
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv, ReviewEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)


def test_parent_edits_do_not_revoke_agreement_or_worker(reserved: ParentEnv) -> None:
    p = reserved
    run(p)
    before = task(p)
    data = p.state()
    data["issues"]["1"]["body"] += "\n\n## 計画\n確認の順序を変える。\n"
    data["issues"]["10"]["body"] += "\n\n## 補足\n証拠の読み方。\n"
    p.save(data)
    status = json.loads(p.cli("status").stdout)
    assert status["input_changes"]["body_changed"]
    assert task(p) == before  # Observation must not rewrite approval/history.
    run(p, "--resume")
    assert task(p)["delegation_id"] == before["delegation_id"]
    assert len(task(p)["worker_runs"]) == 2


def test_default_work_and_postmerge_resume_need_no_repair_file(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p)
    assert task(p)["preflight"]["phase"] == "implementing"
    first = latest(p)
    ended(p)
    data = p.state()
    data["issues"]["10"].update(state="closed", state_reason="completed")
    pr = p.pr(100, 10, branch="codex/worker")
    pr.update(state="closed", merged=True, merged_at="2026-11-08T00:00:00Z")
    data["prs"] = [pr]
    p.save(data)
    result = json.loads(run(p, "--resume").stdout)
    assert result["final_file"] == str(latest(p) / "final.md")
    assert first != latest(p)
    calls = [
        json.loads(line)
        for line in (p.root / "runner-calls.jsonl").read_text().splitlines()
    ]
    assert str(first) in calls[-1]


def test_parent_can_continue_an_investigation_with_direct_instructions(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p, "--planning")
    assert task(p)["preflight"]["phase"] == "planning"
    run(p, "--resume")
    assert task(p)["preflight"]["phase"] == "implementing"


def test_missing_phase_history_does_not_require_fabricated_approval(
    reserved: ParentEnv,
) -> None:
    from tests.e2e.agents_ops.test_manual_worker_runs import store

    p = reserved
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["10"].pop("preflight")
        registry.save()
    run(p)
    value = task(p)["preflight"]
    assert value["phase"] == "implementing"
    assert value["decisions"] == []


@pytest.mark.parametrize("after_planning", [False, True])
def test_failed_prelaunch_can_be_retried_without_a_phantom_run(
    reserved: ParentEnv,
    after_planning: bool,
) -> None:
    p = reserved
    if after_planning:
        run(p, "--planning")
        manual(
            p,
            "decide-worker-plan",
            "--child",
            "10",
            "--action",
            "approve",
            "--answer-file",
            p.note("既存の根拠で実装を進める。"),
        )
    command = (
        ("run-worker", "--child", "10", "--resume")
        if after_planning
        else ("run-worker", "--child", "10", "--prompt-file", p.note())
    )
    # A known prelaunch rejection is retained as an attempt, not a worker run.
    failed = manual(p, *command, ok=False, MANUAL_ADMISSION_FAILURE="1")
    assert len(task(p)["worker_runs"]) == int(after_planning)
    assert json.loads(failed.stdout)["log_publication"]["stage"] == "not_started"
    resumed = manual(p, *command)
    assert len(task(p)["worker_runs"]) == int(after_planning) + 1
    assert json.loads(resumed.stdout)["log_publication"]["stage"] == "receipt_verified"
    assert task(p)["preflight"]["turns"][-2]["cancelled"]
    assert len(task(p)["preflight"]["turns"]) == int(after_planning) + 2
    calls = (p.root / "runner-calls.jsonl").read_bytes()
    # This fixture's repository deliberately cannot publish; every journal must
    # still be visited and reported instead of aborting on the rejected attempt.
    publication = json.loads(
        manual(p, "publish-worker-log", "--child", "10", "--retry", ok=False).stdout
    )
    by_sequence = {item["sequence"]: item for item in publication["logs"]}
    assert by_sequence[int(after_planning) + 1]["stage"] == "not_started"
    assert by_sequence[int(after_planning) + 2]["stage"] == "receipt_verified"
    assert (p.root / "runner-calls.jsonl").read_bytes() == calls


def test_handoff_preserves_review_but_not_old_owners_authority(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    saved = (g.review.run_dir / "result.json").read_bytes()
    p.cli("handoff", "--note-file", p.note("成果を引き継ぐ。"))
    old_identity = p.env["CLAUDE_CODE_SESSION_ID"]
    p.env["CLAUDE_CODE_SESSION_ID"] = "new-owner-session"
    p.cli("handoff", "--accept", "--note-file", p.note("対象と証拠を確認した。"))
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert all(row["decision"] == "merge" for row in rows.values())
    assert (g.review.run_dir / "result.json").read_bytes() == saved
    assert len(g.review.calls()) == 1
    assert g.cli("merge", CLAUDE_CODE_SESSION_ID=old_identity).returncode == 2
    assert not g.merge_calls()


def test_prepared_review_can_be_executed_by_the_new_owner(
    single_review_env: ReviewEnv,
) -> None:
    r, p = single_review_env, single_review_env.p
    request = r.request_path.read_bytes()
    p.cli("handoff", "--note-file", p.note("準備したレビューを引き継ぐ。"))
    p.env["CLAUDE_CODE_SESSION_ID"] = "new-review-owner"
    p.cli("handoff", "--accept", "--note-file", p.note("レビュー対象を確認した。"))
    r.run()
    execution = json.loads((r.run_dir / "execution.json").read_text())
    assert execution["authorization"] == r.request["authorization"]
    assert execution["operator_authorization"]["generation"] == 2
    assert r.request_path.read_bytes() == request
    r.run()
    assert len(r.calls()) == 1


def test_parent_can_verify_artifacts_from_an_ended_failed_turn(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p, ok=False, MANUAL_RUNNER_EXIT="7")
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/worker")]
    p.save(state)
    manual(
        p,
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("失敗の内容と実PRを確認し、成果の範囲を個別に検証した。"),
    )
    assert task(p)["verification"]["prs"] == [100]
    assert json.loads((latest(p) / "result.json").read_text())["result"] == "failed"


def test_closed_parent_continuation_does_not_authorize_creating_new_children(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    state = p.state()
    state["issues"]["1"].update(state="closed", state_reason="completed")
    p.save(state)
    before = p.state()
    result = p.child("new-work", ok=False)
    assert "not open for creating new children" in result.stderr
    assert p.state() == before
