"""Recovery uses real CLIs/git, with private fake GitHub and reviewer evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv, ci_rollup
from tests.e2e.agents_ops.test_parent_session import (
    ReviewEnv,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import pair_review_env as pair_review_env
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)


def json_edit(path: Path, **changes: Any) -> None:
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))


def capture_records(r: ReviewEnv) -> list[dict[str, Any]]:
    capture = importlib.import_module("decision_logs.capture")
    directory = Path(r.p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    value = capture.status(directory, "o/r", 1)
    assert value["status"] == "captured", value
    assert all(e["view_status"] == "ready" for e in value["records"]), value
    return [e["record"] for e in value["records"]]


def next_request(r: ReviewEnv, *, source_run: str | None = None) -> ReviewEnv:
    p = r.p
    manifest = json.loads(
        p.cli(
            "review-prepare",
            *(["--recover-source-run", source_run] if source_run is not None else []),
            "--prs",
            *[str(pr["number"]) for pr in r.request["prs"]],
            "--reason-file",
            p.note("保存済みfinalを検証して採用する。追加AI起動なし。"),
        ).stdout
    )
    request = json.loads(
        p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            p.note("元runの終了・cleanup・出力と対象を照合してrecoverする。"),
        ).stdout
    )
    path = p.root / "recovery-request.json"
    path.write_text(json.dumps(request))
    return replace(r, request=request, request_path=path)


def recovery_request(r: ReviewEnv) -> ReviewEnv:
    return next_request(r, source_run=r.request["review_run_id"])


def recover(r: ReviewEnv, adopted: ReviewEnv, *, ok: bool = True) -> Any:
    return adopted.cli(
        "recover",
        "--request",
        str(adopted.request_path),
        "--source-run",
        r.request["review_run_id"],
        "--reason-file",
        r.p.note("runtime障害後もfinalが有効であることを確認した。"),
        ok=ok,
    )


def gate_for(r: ReviewEnv, monkeypatch: pytest.MonkeyPatch) -> GateEnv:
    state = r.p.state()
    for pr in state["prs"]:
        pr.update(labels=[], mergeable=True, statusCheckRollup=ci_rollup("success"))
    r.p.save(state)
    return GateEnv(r, review_policy_module(monkeypatch))


@pytest.mark.parametrize("failure", ["runtime", "postcheck"])
def test_recover_preserves_failed_source_and_publishes_without_agent(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    r = single_review_env
    output = json.loads(r.output.read_text())
    output["verdict"] = "request_changes"
    output["prs"][0]["findings"] = [
        {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "正しさ",
            "detail": "静的確認: 保存後の復旧で指摘を保持する。",
        }
    ]
    r.output.write_text(json.dumps(output))
    response = json.loads(
        r.run(
            ok=False,
            extra={"REVIEW_AGENT_FINAL_EXIT": "3", "REVIEW_FINAL_ONLY": "1"}
            if failure == "runtime"
            else {"REVIEW_POSTCHECK_FAIL": "1"},
        ).stdout
    )
    assert response["result"]["status"] == "failed"
    execution = json.loads((r.run_dir / "execution.json").read_text())
    assert execution["stages"]["agent"]["cleanup_complete"] is True
    assert execution["stages"]["output"]["status"] == "validated"
    assert execution["stages"]["postcheck"]["status"] == (
        "passed" if failure == "runtime" else "failed"
    )
    original = {
        str(p.relative_to(r.run_dir)): p.read_bytes()
        for p in r.run_dir.rglob("*")
        if p.is_file() and p.name != "run.lock"
    }
    source_comments = copy.deepcopy(r.comments(1))
    g = gate_for(r, monkeypatch)
    adopted = recovery_request(r)
    value = json.loads(recover(r, adopted).stdout)
    result = value["result"]
    assert value["agent_invocations"] == 0 and value["publication"] == "published"
    assert result["kind"] == "adoption" and result["mode"] == "recover"
    assert result["source_run"] == r.request["review_run_id"]
    assert result["prs"][0]["findings"][0]["id"] == "pr-100-1"
    assert len(r.calls()) == 1
    for name, raw in original.items():
        assert (r.run_dir / name).read_bytes() == raw
    assert r.comments(1)[: len(source_comments)] == source_comments
    adopted_execution = json.loads((adopted.run_dir / "execution.json").read_text())
    assert adopted_execution["agent_invocations"] == 0
    captured = capture_records(r)
    actions = {e["id"]: e for e in captured if e["kind"] == "action"}
    recovered = [
        e
        for e in captured
        if e["kind"] == "result"
        and "review.recover: agent_invocations=0" in e["data"]["summary"]
    ]
    assert (
        recovered
        and actions[recovered[-1]["data"]["action_id"]]["data"]["operation"]
        == "retry_fetch"
    )
    assert recovered[-1]["data"]["previous_result_id"] is not None
    assert recovered[-1]["data"]["auxiliary"]["usage"]["status"] == "unavailable"
    g = replace(g, review=adopted)
    g.confirm()
    blocked = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]
    assert "unresolved_findings" in blocked["reasons"]
    assert json.loads(recover(r, adopted).stdout)["result"] == result
    worktree = r.p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("fixed after adopting the saved review\n")
    run_git(worktree, "commit", "-am", "fix recovered finding")
    fixed_head = run_git(worktree, "rev-parse", "HEAD")
    state = r.p.state()
    state["prs"][0]["head"]["sha"] = fixed_head
    r.p.save(state)
    proof = {
        "ci": "同HEAD Python成功",
        "regression_tests": "関連回帰テスト成功",
        "resolution": "親が指摘と当該コードを確認",
        "reviewed_paths": ["change.txt"],
    }
    evidence = r.p.root / "fix-proof.json"
    evidence.write_text(json.dumps(proof))
    adopted.cli(
        "resolve",
        "--run",
        result["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "pr-100-1",
        "--fixed-head",
        fixed_head,
        "--evidence",
        str(evidence),
    )
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"] == "merge"
    captured = capture_records(r)
    verified = [
        e
        for e in captured
        if e["kind"] == "result"
        and "review.resolve: explicit verification recorded" in e["data"]["summary"]
    ]
    assert (
        verified[-1]["data"]["observer"] == "parent_verification"
        and verified[-1]["data"]["verifies"]
    )
    # An accidental run after the fix never launches a second agent either.
    adopted.run(ok=False)
    assert len(r.calls()) == 1


@pytest.mark.parametrize(
    "case",
    [
        "json",
        "target",
        "verdict",
        "digest",
        "cleanup",
        "workspace_cleanup",
        "runtime_run",
        "runtime_json",
        "current_target",
    ],
)
def test_recovery_rejects_unverified_final_without_writes_or_agent(
    single_review_env: ReviewEnv, case: str
) -> None:
    r = single_review_env
    if case in {"json", "target", "verdict"}:
        output = json.loads(r.output.read_text())
        if case == "target":
            output["prs"][0]["number"] = 999
        elif case == "verdict":
            output["verdict"] = "incomplete"
        r.output.write_text("{invalid" if case == "json" else json.dumps(output))
    r.run(
        ok=False,
        extra={
            "REVIEW_AGENT_FINAL_EXIT": "3",
            **({"REVIEW_BROKEN_RUNTIME": "1"} if case == "runtime_json" else {}),
        },
    )
    if case == "runtime_json":
        execution = json.loads((r.run_dir / "execution.json").read_text())
        assert execution["stages"]["output"]["status"] == "invalid"
        assert execution["stages"]["agent"]["status"] == "unconfirmed"
        assert (
            r.run_dir / "agent-runtime/final.md"
        ).read_text() == r.output.read_text()
    adopted = recovery_request(r)
    if case == "digest":
        json_edit(r.run_dir / "execution.json", request_sha256="0" * 64)
    elif case == "cleanup":
        json_edit(r.run_dir / "agent-runtime/result.json", cleanup_complete=False)
    elif case == "workspace_cleanup":
        execution = json.loads((r.run_dir / "execution.json").read_text())
        execution["stages"]["agent"]["workspace_cleanup_complete"] = False
        (r.run_dir / "execution.json").write_text(json.dumps(execution))
    elif case == "runtime_run":
        json_edit(r.run_dir / "agent-runtime/result.json", prompt_sha256="0" * 64)
    elif case == "current_target":
        state = r.p.state()
        state["prs"][0]["head"]["sha"] = state["prs"][0]["base"]["sha"]
        r.p.save(state)
    before = copy.deepcopy(r.p.state()["comments"])
    response = recover(r, adopted, ok=False)
    assert response.returncode == 2
    assert not (adopted.run_dir / "result.json").exists()
    assert r.p.state()["comments"] == before and len(r.calls()) == 1
    captured = capture_records(r)
    actions = {e["id"]: e for e in captured if e["kind"] == "action"}
    rejected = [
        e
        for e in captured
        if e["kind"] == "result"
        and "review.recover: post-start failure" in e["data"]["summary"]
    ]
    assert (
        rejected
        and actions[rejected[-1]["data"]["action_id"]]["data"]["operation"]
        == "retry_fetch"
    )
    assert rejected[-1]["data"]["previous_result_id"] is not None
    assert rejected[-1]["data"]["status"] == "failure"
    assert rejected[-1]["data"]["observer"] == "machine"
    assert not rejected[-1]["data"]["verifies"]


def inject_lost_response(r: ReviewEnv, *, target: str, method: str) -> None:
    gh = r.p.root / "bin/gh"
    source = gh.read_text()
    # The fake commits its write first, then drops the response exactly once.
    source = source.replace(
        "def response(value, code=200):",
        "def response(value, code=200):\n"
        f'    if globals().get("path") == {target!r} and globals().get("method") == {method!r} and not state.get("lost_response"):\n'
        '        state["lost_response"] = True\n        save()\n        fail()',
    )
    gh.write_text(source)


@pytest.mark.parametrize(
    "failure",
    [
        "pending",
        "projection",
        "lost_pending",
        "lost_projection",
        "lost_result",
        "interrupted_pending",
    ],
)
def test_publication_resume_reconciles_each_write_without_duplicates(
    pair_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    r = pair_review_env
    extra = {}
    if failure == "pending":
        extra["PARENT_FAIL_PATH"] = "repos/o/r/issues/1/comments"
    elif failure == "projection":
        extra["PARENT_FAIL_PATH"] = "repos/o/r/issues/101/comments"
    elif failure.startswith("lost") or failure == "interrupted_pending":
        target = (
            "repos/o/r/issues/1/comments"
            if failure in {"lost_pending", "interrupted_pending"}
            else "repos/o/r/issues/101/comments"
        )
        if failure == "lost_result":
            next_id = 1 + max(
                c["id"] for rows in r.p.state()["comments"].values() for c in rows
            )
            target = f"repos/o/r/issues/comments/{next_id}"
        inject_lost_response(
            r, target=target, method="PATCH" if failure == "lost_result" else "POST"
        )
    response = json.loads(r.run(ok=False, extra=extra).stdout)
    assert response["result"]["status"] == "succeeded"
    assert response["publication"] == "publication_failed"
    if failure == "interrupted_pending":
        # SIGKILL after the pending write leaves the aggregate status running.
        execution = json.loads((r.run_dir / "execution.json").read_text())
        execution["status"] = "running"
        execution["stages"]["publication"]["status"] = "pending"
        (r.run_dir / "execution.json").write_text(json.dumps(execution))
    immutable = (r.run_dir / "result.json").read_bytes()
    resumed = json.loads(
        r.cli("resume-publication", "--run", r.request["review_run_id"]).stdout
    )
    assert resumed["publication"] == "published" and resumed["agent_invocations"] == 0
    assert (r.run_dir / "result.json").read_bytes() == immutable
    policy = review_policy_module(monkeypatch)
    for n, kind in ((1, "result"), (100, "pr"), (101, "pr")):
        rows = [
            record
            for record in policy.records(r.comments(n), "operator")
            if record["review_run_id"] == r.request["review_run_id"]
        ]
        assert len(rows) == 1 and rows[0]["kind"] == kind
    before = copy.deepcopy(r.p.state()["comments"])
    r.cli("resume-publication", "--run", r.request["review_run_id"])
    assert r.p.state()["comments"] == before and len(r.calls()) == 1
    captured = capture_records(r)
    actions = {e["id"]: e for e in captured if e["kind"] == "action"}
    resumed_results = [
        e
        for e in captured
        if e["kind"] == "result"
        and "review.resume-publication: agent_invocations=0" in e["data"]["summary"]
    ]
    assert (
        resumed_results
        and actions[resumed_results[-1]["data"]["action_id"]]["data"]["operation"]
        == "retry_publication"
    )
    assert resumed_results[-1]["data"]["previous_result_id"] is not None


def test_evidence_wait_requires_current_four_key_public_completion(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    output = json.loads(r.output.read_text())
    output["prs"][0]["evidence_waits"] = [
        {
            "question": "境界条件のCPUテスト結果",
            "required_evidence": "固定HEADで既存テストが成功すること",
        }
    ]
    r.output.write_text(json.dumps(output))
    r.run()
    g = gate_for(r, monkeypatch)
    g.confirm()
    assert (
        "pr_100_evidence_1_pending"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )
    snapshot = r.request["prs"][0]
    paths = run_git(
        r.p.repo,
        "diff",
        "--name-only",
        snapshot["merge_base_sha"],
        snapshot["head_sha"],
    ).splitlines()
    proof = {
        "ci": "固定HEADのPython成功",
        "regression_tests": "既存境界テスト成功を親が確認",
        "resolution": "必要な結果を取得して論点を確認した",
        "reviewed_paths": paths,
    }
    evidence = r.p.root / "wait-proof.json"
    evidence.write_text(
        json.dumps({k: v for k, v in proof.items() if k != "reviewed_paths"})
    )
    args = (
        "complete-evidence",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--wait",
        "1",
        "--evidence",
        str(evidence),
    )
    r.cli(*args, ok=False)
    assert (
        "pr_100_evidence_1_pending"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )
    evidence.write_text(json.dumps(proof))
    r.cli(*args)
    captured = capture_records(r)
    verified = [
        e
        for e in captured
        if e["kind"] == "result"
        and "review.complete-evidence: agent_invocations=0" in e["data"]["summary"]
    ]
    assert (
        verified[-1]["data"]["observer"] == "parent_verification"
        and verified[-1]["data"]["verifies"]
    )
    before = copy.deepcopy(r.p.state()["comments"])
    r.cli(*args)
    assert r.p.state()["comments"] == before and len(r.calls()) == 1
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"] == "merge"
    # One public copy is insufficient, even when the local receipt says published.
    state = r.p.state()
    state["comments"]["100"] = [
        c for c in state["comments"]["100"] if '"kind":"evidence"' not in c["body"]
    ]
    r.p.save(state)
    assert (
        "pr_100_evidence_1_pending"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )


@pytest.mark.parametrize(
    "case",
    [
        "source_missing",
        "unregistered_source",
        "source_invocations",
        "adoption_invocations",
        "unpublished",
        "same_run_failed",
        "latest_ci",
    ],
)
def test_adoption_gate_checks_real_source_publication_and_current_ci(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    r = single_review_env
    r.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3"})
    g = gate_for(r, monkeypatch)
    adopted = recovery_request(r)
    result = json.loads(recover(r, adopted).stdout)["result"]
    g = replace(g, review=adopted)
    g.confirm()
    if case == "source_missing":
        (r.run_dir / "execution.json").unlink()
    elif case == "unregistered_source":
        registry_path = r.run_dir.parents[1] / "registry.json"
        registry = json.loads(registry_path.read_text())
        registry["parents"]["o/r#1"]["reviews"] = [
            row
            for row in registry["parents"]["o/r#1"]["reviews"]
            if row["request"]["review_run_id"] != r.request["review_run_id"]
        ]
        registry_path.write_text(json.dumps(registry))
    elif case == "source_invocations":
        json_edit(r.run_dir / "execution.json", agent_invocations=0)
    elif case == "adoption_invocations":
        json_edit(adopted.run_dir / "execution.json", agent_invocations=1)
    elif case == "unpublished":
        json_edit(adopted.run_dir / "execution.json", status="publication_failed")
    elif case == "same_run_failed":
        state = r.p.state()
        failed = json.loads((r.run_dir / "result.json").read_text())
        failed.update(
            review_run_id=result["review_run_id"],
            request=result["request"],
            request_sha256=result["request_sha256"],
        )
        failed.update(
            {
                k: result[k]
                for k in (
                    "kind",
                    "mode",
                    "source_run",
                    "reason",
                    "agent_invocations",
                    "validation",
                    "runtime_settings",
                )
            }
        )
        g.policy.validate_result(failed)
        state["comments"]["1"].append(
            {
                "id": 10000,
                "user": {"login": "operator"},
                "body": g.policy.marker(failed),
            }
        )
        r.p.save(state)
    elif case == "latest_ci":
        state = r.p.state()
        state["prs"][0]["statusCheckRollup"] = ci_rollup("failure")
        r.p.save(state)
    response = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert response.returncode in {1, 2}
    assert len(r.calls()) == 1 and not g.merge_calls()
    if case == "same_run_failed":
        assert response.returncode == 2
        error = json.loads(response.stderr)
        assert error["reasons"] == ["input_or_collection_error"]
        assert "conflicting parent evidence record" in error["error"]
    elif case in {"source_missing", "source_invocations", "unregistered_source"}:
        assert response.returncode == 2
        if case == "unregistered_source":
            assert "registered source review" in response.stderr
    else:
        reasons = g.results(response)[100]["reasons"]
        expected = (
            "ci_python_failure"
            if case == "latest_ci"
            else "review_not_published_or_authorized"
        )
        assert expected in reasons


def test_legacy_run_without_stages_or_evidence_waits_can_be_recovered(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    r.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3"})
    execution = json.loads((r.run_dir / "execution.json").read_text())
    del execution["stage_version"], execution["stages"]
    (r.run_dir / "execution.json").write_text(json.dumps(execution))
    (r.run_dir / "output.json").unlink()
    output = json.loads(r.output.read_text())
    del output["prs"][0]["evidence_waits"]
    for final in ("agent_output.md", "agent-runtime/final.md"):
        (r.run_dir / final).write_text(json.dumps(output))
    events_path = r.run_dir / "agent-runtime/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    next(event for event in events if event["type"] == "item.completed")["item"][
        "text"
    ] = json.dumps(output)
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    schema_path = r.run_dir / "agent-runtime/output-schema.json"
    schema = json.loads(schema_path.read_text())
    pr_schema = schema["properties"]["prs"]["items"]
    del pr_schema["properties"]["evidence_waits"]
    pr_schema["required"].remove("evidence_waits")
    schema_path.write_text(json.dumps(schema))
    runtime_path = r.run_dir / "agent-runtime/result.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["contract"]["json_schema_sha256"] = hashlib.sha256(
        schema_path.read_bytes()
    ).hexdigest()
    runtime_path.write_text(json.dumps(runtime))
    # Legacy stages cannot establish exit for cross-run exclusion. Confirm only
    # cleanup; the original failed result and recovery contract remain unchanged.
    original_result = (r.run_dir / "result.json").read_bytes()
    r.cli(
        "close-interrupted",
        "--run",
        r.request["review_run_id"],
        "--reason-file",
        r.p.note("旧形式runのruntime cleanupとworkspace撤去を確認した。"),
    )
    assert (r.run_dir / "result.json").read_bytes() == original_result
    g = gate_for(r, monkeypatch)
    adopted = recovery_request(r)
    result = json.loads(recover(r, adopted).stdout)["result"]
    assert result["status"] == "succeeded" and "evidence_waits" not in result["prs"][0]
    g = replace(g, review=adopted)
    g.confirm()
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"] == "merge"
    assert len(r.calls()) == 1


def newer_findings(r: ReviewEnv) -> ReviewEnv:
    newer = next_request(r)
    output = json.loads(newer.output.read_text())
    output["verdict"] = "request_changes"
    output["prs"][0]["findings"] = [
        {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "後続レビュー",
            "detail": "Aが見落とした未解決の問題",
        }
    ]
    newer.output.write_text(json.dumps(output))
    newer.run()
    return newer


def resolve_finding(r: ReviewEnv) -> None:
    proof = r.p.root / "later-finding-proof.json"
    proof.write_text(
        json.dumps(
            {
                "ci": "固定HEAD Python成功",
                "regression_tests": "親が回帰結果を確認",
                "resolution": "後続runの指摘を確認して解消",
                "reviewed_paths": [],
            }
        )
    )
    r.cli(
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "pr-100-1",
        "--fixed-head",
        r.request["prs"][0]["head_sha"],
        "--evidence",
        str(proof),
    )


def test_recover_cannot_skip_later_findings(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = single_review_env
    a.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3"})
    b = newer_findings(a)
    c = recovery_request(b)
    rejected = recover(a, c, ok=False)
    assert rejected.returncode == 2 and "source differs from frozen" in rejected.stderr
    assert not (c.run_dir / "result.json").exists()
    # The latest source retains B's major, including through an adoption.
    result = json.loads(recover(b, c).stdout)["result"]
    assert result["verdict"] == "request_changes"
    g = gate_for(c, monkeypatch)
    g.confirm()
    assert (
        "unresolved_findings"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )
    resolve_finding(c)
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"] == "merge"
    assert len(a.calls()) == 2


def test_gate_and_resume_reject_legacy_laundered_adoption(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = single_review_env
    a.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3"})
    b = newer_findings(a)
    c = recovery_request(a)
    registry_path = a.run_dir.parents[1] / "registry.json"
    original = registry_path.read_bytes()
    registry = json.loads(original)
    # Reproduce an already-published C from before the history guard: its creation
    # did not see B. Restore the complete durable history before invoking gate.
    registry["parents"]["o/r#1"]["reviews"] = [
        row
        for row in registry["parents"]["o/r#1"]["reviews"]
        if row["request"]["review_run_id"] != b.request["review_run_id"]
    ]
    registry_path.write_text(json.dumps(registry))
    try:
        recover(a, c)
    finally:
        registry_path.write_bytes(original)
    g = gate_for(c, monkeypatch)
    g.confirm()
    assert (
        "adoption_source_superseded"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )
    comments = copy.deepcopy(a.p.state()["comments"])
    resume = c.cli("resume-publication", "--run", c.request["review_run_id"], ok=False)
    assert resume.returncode == 2 and "superseded" in resume.stderr
    assert a.p.state()["comments"] == comments
    resolve_finding(b)
    # Resolving B does not turn C's older approve into the latest review evidence.
    assert (
        "adoption_source_superseded"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )
    assert not g.merge_calls() and len(a.calls()) == 2


@pytest.mark.parametrize(
    "failure", ["event_mismatch", "invalid_output", "metadata_error"]
)
def test_recover_refuses_inconsistent_runner_evidence(
    single_review_env: ReviewEnv, failure: str
) -> None:
    r = single_review_env
    extra = {"REVIEW_AGENT_FINAL_EXIT": "3", "REVIEW_FINAL_ONLY": "1"}
    if failure == "event_mismatch":
        event_output = json.loads(r.output.read_text())
        event_output["verdict"] = "request_changes"
        event_output["prs"][0]["findings"] = [
            {
                "severity": "major",
                "path": "change.txt",
                "line": 1,
                "rule": "正しさ",
                "detail": "完了eventはmajorを報告している",
            }
        ]
        path = r.p.root / "event-output.json"
        path.write_text(json.dumps(event_output))
        extra["REVIEW_EVENT_OUTPUT"] = str(path)
    else:
        extra["REVIEW_RUNTIME_REASON"] = failure
    response = json.loads(r.run(ok=False, extra=extra).stdout)
    assert response["result"]["status"] == "failed"
    original_final = (r.run_dir / "agent-runtime/final.md").read_bytes()
    c = recovery_request(r)
    before = copy.deepcopy(r.p.state()["comments"])
    assert recover(r, c, ok=False).returncode == 2
    assert not (c.run_dir / "result.json").exists()
    assert r.p.state()["comments"] == before
    assert (r.run_dir / "agent-runtime/final.md").read_bytes() == original_final
    assert len(r.calls()) == 1


def test_gate_rechecks_event_final_correspondence(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = single_review_env
    a.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3"})
    c = recovery_request(a)
    recover(a, c)
    g = gate_for(c, monkeypatch)
    g.confirm()
    events_path = a.run_dir / "agent-runtime/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    next(e for e in events if e["type"] == "item.completed")["item"]["text"] = (
        '{"verdict":"request_changes"}'
    )
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    response = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert (
        response.returncode == 2
        and "completed agent message disagree" in response.stderr
    )
    assert not g.merge_calls() and len(a.calls()) == 1
