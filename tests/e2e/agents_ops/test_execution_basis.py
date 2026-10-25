"""Execution evidence through fake GitHub/runtime and private disposable state."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    make_review_env,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def registry(p: ParentEnv) -> Path:
    return (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )


def spec(p: ParentEnv, ignored: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    policy = importlib.import_module("parent.policy")
    args = ("--ignored-comments", p.spec(ignored)) if ignored else ()
    inputs = json.loads(p.cli("basis-input", *args).stdout)
    return {
        "basis": {
            "schema_version": 1,
            "kind": "autonomous_decision",
            "purpose": "未担当の具体的な作業を進める",
            "scope": "issueに記載した変更範囲と対象外を守る",
            "behavior": "共通APIから委任とレビューを要求する",
            "acceptance": "fake CLIと一時stateで境界を確認する",
            "rationale": "issueと関連コードから実装範囲と完了条件を確定できる",
            "sources": ["https://github.com/o/r/issues/1", "code:fixed-commit:path"],
            "input_sha256": policy.digest(inputs),
            "policy_version": "autonomous-start/v1",
        },
        "inputs": inputs,
        "ignored_comments": ignored or [],
    }


def record(p: ParentEnv, value: dict[str, Any] | None = None) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(p.cli("basis-record", "--spec", p.spec(value or spec(p))).stdout),
    )


def autonomous(p: ParentEnv) -> None:
    p.cli("register")
    record(p)


def test_valid_autonomous_basis_allows_split_delegation_and_result(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    p.child("unrecorded", ok=False)
    value = spec(p)
    recorded = record(p, value)
    record(p, value)
    assert (
        sum(
            '"event": "execution_basis"' in c["body"]
            for c in p.state()["comments"]["1"]
        )
        == 1
    )
    assert not any('"event": "agreed"' in c["body"] for c in p.state()["comments"]["1"])
    p.child("worker")
    worktree = p.worktree("worker")
    launch = json.loads(
        p.cli(
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/worker",
            "--worktree",
            str(worktree),
        ).stdout
    )
    auth = launch["assignment"]["authorization"]
    assert auth == {
        "generation": 1,
        "execution_basis_kind": "autonomous_decision",
        "execution_basis_sha256": recorded["basis_sha256"],
    }
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/worker")]
    p.save(state)
    p.cli("record-result", "--child", "10", "--evidence-file", p.note())
    row = json.loads(registry(p).read_text())["parents"]["o/r#1"]
    assert row["tasks"]["10"]["verification"]["authorization"] == auth


@pytest.mark.parametrize(
    "case",
    [
        "version",
        "policy",
        "human",
        "digest",
        "empty",
        "input",
        "input-schema",
        "unknown",
    ],
)
def test_invalid_autonomous_basis_has_no_write(
    parent_env: ParentEnv, case: str
) -> None:
    p = parent_env
    p.cli("register")
    value = spec(p)
    if case == "version":
        value["basis"]["schema_version"] = 99
    elif case == "policy":
        value["basis"]["policy_version"] = "unreviewed/v2"
    elif case == "human":
        value["basis"]["human_confirmation"] = "generated confirmation"
    elif case == "digest":
        value["basis"]["input_sha256"] = "f" * 64
    elif case == "empty":
        value["basis"]["rationale"] = " "
    elif case == "input":
        value["inputs"]["issue"]["body_sha256"] = "a" * 64
        value["basis"]["input_sha256"] = importlib.import_module(
            "parent.policy"
        ).digest(value["inputs"])
    elif case == "input-schema":
        value["inputs"]["schema_version"] = 99
        value["basis"]["input_sha256"] = importlib.import_module(
            "parent.policy"
        ).digest(value["inputs"])
    else:
        value["unexpected"] = True
    before = p.state(), registry(p).read_bytes()
    p.cli("basis-record", "--spec", p.spec(value), ok=False)
    assert (p.state(), registry(p).read_bytes()) == before
    p.child("invalid", ok=False)


def test_unassigned_record_and_owner_generation_handoff(parent_env: ParentEnv) -> None:
    p = parent_env
    candidate = record(p)
    assert candidate["generation"] == 1
    args = (
        "basis-check",
        "--basis-sha256",
        candidate["basis_sha256"],
        "--generation",
        "1",
    )
    assert (
        json.loads(p.cli(*args).stdout)["execution_basis_sha256"]
        == candidate["basis_sha256"]
    )
    assert not registry(p).exists()  # A judgment does not claim a parent.
    p.cli("register")
    p.cli(*args, ok=False)  # A manual registration wins over a stale candidate.
    p.child("existing")
    value = spec(p)
    second = {"CLAUDE_CODE_SESSION_ID": "private-other-session"}
    p.cli("basis-record", "--spec", p.spec(value), extra=second, ok=False)
    p.cli("handoff", "--note-file", p.note())
    p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    p.child("stale-owner", ok=False)
    p.env.update(second)
    p.child("stale-basis", ok=False)
    assert record(p)["generation"] == 2
    p.child("next")
    assert "10" in json.loads(registry(p).read_text())["parents"]["o/r#1"]["tasks"]


def test_candidate_rechecks_inputs_without_creating_assignment(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    candidate = record(p)
    state = p.state()
    state["issues"]["1"]["body"] += "\n\n## 指示\n変更された範囲。\n"
    p.save(state)
    p.cli(
        "basis-check",
        "--basis-sha256",
        candidate["basis_sha256"],
        "--generation",
        "1",
        ok=False,
    )
    assert not registry(p).exists()


def test_judgment_job_needs_no_runtime_identity_but_rejects_ambiguous_identity(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    value = spec(p)
    p.cli(
        "basis-record",
        "--spec",
        p.spec(value),
        extra={"CLAUDE_CODE_SESSION_ID": "", "CODEX_SESSION_ID": ""},
    )
    before = p.state()
    p.cli(
        "basis-record",
        "--spec",
        p.spec(value),
        extra={"CLAUDE_CODE_SESSION_ID": "one", "CODEX_SESSION_ID": "two"},
        ok=False,
    )
    assert p.state() == before


def test_long_issue_and_instruction_are_bound_without_copying_them_into_public_record(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["body"] += "\n\n## 詳細\n" + "具体的な要件。" * 10000
    state["comments"]["1"] = [
        {"id": 900, "user": {"login": "human"}, "body": "重要な追加指示。" * 10000}
    ]
    p.save(state)
    value = spec(p)
    assert "body_sha256" in value["inputs"]["issue"]
    assert "sha256" in value["inputs"]["comments"][0]
    record(p, value)
    assert len(p.state()["comments"]["1"][-1]["body"]) < 60000
    state = p.state()
    state["comments"]["1"][0]["body"] += "停止の指示。"
    p.save(state)
    p.cli(
        "basis-check",
        "--basis-sha256",
        importlib.import_module("parent.policy").digest(value["basis"]),
        "--generation",
        "1",
        ok=False,
    )


@pytest.mark.parametrize(
    "heading,stable",
    [
        ("Acceptance checklist", True),
        ("全体の完了条件", True),
        ("対象外", False),
        ("方針の選択", False),
    ],
)
def test_only_acceptance_completion_checks_are_nonmaterial(
    parent_env: ParentEnv, heading: str, stable: bool
) -> None:
    execution = importlib.import_module("parent.execution")
    before = f"## {heading}\n\n- [ ] この項目\n"
    after = before.replace("[ ]", "[x]")
    assert (
        execution.requirement_body(before) == execution.requirement_body(after)
    ) is stable


def test_returning_to_a_previous_basis_and_repeated_revocation_are_not_lost(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    autonomous(p)
    original = spec(p)
    changed = copy.deepcopy(original)
    changed["basis"]["scope"] = "別の範囲"
    record(p, changed)
    restored = record(p, original)
    history = importlib.import_module("parent.policy").events(
        p.state()["comments"]["1"], "operator", 1
    )
    roots = [e for e in history if e["event"] == "execution_basis"]
    assert [e["revision"] for e in roots] == [1, 2, 3]
    assert (
        roots[-1]["basis_sha256"]
        == restored["basis_sha256"]
        == roots[0]["basis_sha256"]
    )
    for _ in range(2):
        p.cli("basis-revoke", "--reason-file", p.note("人間の同じ停止指示"))
        p.cli("basis-revoke", "--reason-file", p.note("人間の同じ停止指示"))
        p.child("revoked", ok=False)
        p.agreed()
    history = importlib.import_module("parent.policy").events(
        p.state()["comments"]["1"], "operator", 1
    )
    assert [e["revision"] for e in history if e["event"] == "basis_revoked"] == [1, 2]


@pytest.mark.parametrize(
    "change", ["body", "instruction", "checking", "revocation", "needs-human"]
)
def test_material_changes_and_explicit_holds_block_autonomous_work(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    autonomous(p)
    value = spec(p)
    state = p.state()
    if change == "body":
        state["issues"]["1"]["body"] += "\n\n## 追加指示\n変更範囲を修正する。\n"
        p.save(state)
    elif change == "instruction":
        state["comments"]["1"].append(
            {
                "id": 900,
                "user": {"login": "human"},
                "body": "この作業は停止し、追加の変更は見送ってください。",
            }
        )
        p.save(state)
    elif change == "checking":
        p.cli("checking")
    elif change == "revocation":
        p.cli(
            "basis-revoke",
            "--reason-file",
            p.note("人間の明示的な見送り。再合意まで停止。"),
        )
    else:
        state["issues"]["1"]["labels"].append({"name": "needs-human"})
        p.save(state)
    p.child("blocked", ok=False)
    before = p.state()
    p.cli("basis-record", "--spec", p.spec(value), ok=False)
    assert p.state() == before
    if change in {"checking", "revocation"}:
        p.cli("basis-record", "--spec", p.spec(spec(p)), ok=False)
        p.agreed()  # Only the existing explicit human agreement path lifts this hold.
        p.child("human-resumed")


def test_progress_and_classified_unrelated_comments_keep_basis_identity(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    autonomous(p)
    original = record(p)
    p.child("first")  # Changes the child list and progress, not requirements.
    policy = importlib.import_module("parent.policy")
    state = p.state()
    state["issues"]["1"]["body"] = policy.replace_section(
        state["issues"]["1"]["body"], "進捗概要", "検証中。"
    )
    unrelated = {"id": 900, "user": {"login": "human"}, "body": "表示を確認しました。"}
    state["comments"]["1"].append(unrelated)
    p.save(state)
    p.child("unclassified", ok=False)
    execution = importlib.import_module("parent.execution")
    ignored = [
        {
            "id": 900,
            "sha256": policy.digest(execution.comment_snapshot(unrelated)),
            "reason": "要件の変更を含まない表示確認",
        }
    ]
    assert record(p, spec(p, ignored))["basis_sha256"] == original["basis_sha256"]
    p.child("after-observation")
    state = p.state()
    # Find the observed comment rather than depending on publication count.
    next(c for c in state["comments"]["1"] if c["id"] == 900)["body"] = (
        "停止してください。"
    )
    p.save(state)
    p.child("edited-observation", ok=False)


def auto_gate(
    p: ParentEnv, monkeypatch: pytest.MonkeyPatch, *, shared: bool = False
) -> GateEnv:
    monkeypatch.setattr(p, "agreed", lambda: autonomous(p))
    r = make_review_env(p, independent=True)
    policy = review_policy_module(monkeypatch)
    if shared:
        worktree = p.repo / ".claude/worktrees/pr-0"
        (worktree / ".agents").mkdir()
        (worktree / ".agents/shared.txt").write_text("shared foundation\n")
        run_git(worktree, "add", ".agents/shared.txt")
        run_git(worktree, "commit", "-m", "shared changes")
        state = p.state()
        state["prs"][0]["head"]["sha"] = run_git(worktree, "rev-parse", "HEAD")
        p.save(state)
        manifest = json.loads(
            p.cli(
                "review-prepare",
                "--prs",
                "100",
                "101",
                "102",
                "--reason-file",
                p.note(),
            ).stdout
        )
        r.request = json.loads(
            p.cli(
                "review-request",
                "--run-id",
                manifest["review_run_id"],
                "--instruction-file",
                p.note(),
            ).stdout
        )
        r.request_path.write_text(json.dumps(r.request))
    state = p.state()
    for pr in state["prs"]:
        pr.update(
            labels=[],
            mergeable=True,
            statusCheckRollup=[
                {
                    "__typename": "CheckRun",
                    "name": "Python",
                    "workflowName": "CI",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
        )
    p.save(state)
    r.run()
    gate = GateEnv(r, policy)
    gate.confirm()
    return gate


def test_autonomous_shared_merge_uses_current_review_without_path_permission(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    g = auto_gate(p, monkeypatch, shared=True)
    request = g.review.request
    assert request["schema_version"] == 2
    assert request["authorization"]["execution_basis_kind"] == "autonomous_decision"
    allowed = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert allowed[100]["decision"] == "merge"
    assert "shared_exemption" not in allowed[100]
    assert g.merge_calls() == []
    assert not any('"event": "agreed"' in c["body"] for c in p.state()["comments"]["1"])
    assert len(g.review.calls()) == 1


def test_autonomous_merge_keeps_basis_valid_through_progress_updates(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = auto_gate(parent_env, monkeypatch)
    response = g.cli("merge", PARENT_MERGE_ADVANCE_BASE="1")
    assert response.returncode == 0, response.stderr + response.stdout
    assert len(g.merge_calls()) == 3
    assert all(row["action"] == "merged" for row in g.results(response).values())


def test_new_basis_does_not_reauthorize_old_review(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    g = auto_gate(p, monkeypatch)
    before = len(g.review.calls())
    value = spec(p)
    value["basis"]["scope"] = "異なる範囲で作業する"
    record(p, value)
    response = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert all(row["decision"] == "blocked" for row in g.results(response).values())
    g.review.cli(
        "run",
        "--request",
        str(g.review.request_path),
        "--agent-exec",
        str(g.review.fake),
        ok=False,
    )
    p.cli(
        "review-state",
        "--run-id",
        g.review.request["review_run_id"],
        "--state",
        "complete",
        "--evidence-file",
        p.note(),
        ok=False,
    )
    assert len(g.review.calls()) == before
    assert g.merge_calls() == []


def test_classifying_unrelated_comment_preserves_existing_review(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    g = auto_gate(p, monkeypatch, shared=True)
    state = p.state()
    comment = {
        "id": 9900,
        "user": {"login": "human"},
        "body": "進捗の表示を確認しました。",
    }
    state["comments"]["1"].append(comment)
    p.save(state)
    policy, execution = (
        importlib.import_module("parent.policy"),
        importlib.import_module("parent.execution"),
    )
    ignored = [
        {
            "id": 9900,
            "sha256": policy.digest(execution.comment_snapshot(comment)),
            "reason": "表示への応答で要件変更なし",
        }
    ]
    assert (
        record(p, spec(p, ignored))["basis_sha256"]
        == g.review.request["authorization"]["execution_basis_sha256"]
    )
    response = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert response.returncode == 0, response.stderr + response.stdout
    assert all(row["decision"] == "merge" for row in g.results(response).values())
    assert len(g.review.calls()) == 1


@pytest.mark.parametrize("change", ["requirements", "basis", "generation"])
def test_stale_delegation_needs_explicit_parent_result_verification(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    autonomous(p)
    p.child("worker")
    worktree = p.worktree("worker")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/worker",
        "--worktree",
        str(worktree),
    )
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/worker")]
    if change == "requirements":
        state["issues"]["10"]["body"] = state["issues"]["10"]["body"].replace(
            "動作する", "追加の境界を確認できる"
        )
    p.save(state)
    if change == "basis":
        value = spec(p)
        value["basis"]["scope"] = "変更後の範囲"
        record(p, value)
    elif change == "generation":
        data = json.loads(registry(p).read_text())
        data["parents"]["o/r#1"]["generation"] += 1
        registry(p).write_text(json.dumps(data))
        record(p)
    p.cli("review-prepare", "--prs", "100", ok=False)
    p.cli(
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("現在の要件・差分・回帰テストを確認して既存成果を採用する。"),
    )
    assert (
        json.loads(p.cli("review-prepare", "--prs", "100").stdout)["schema_version"]
        == 2
    )
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/worker",
        "--worktree",
        str(worktree),
        ok=False,
    )


def test_recording_basis_does_not_resume_a_stopped_child(parent_env: ParentEnv) -> None:
    p = parent_env
    autonomous(p)
    p.child("stopped")
    p.cli("stop", "--child", "10", "--spec", p.detail())
    record(p)
    worktree = p.worktree("stopped")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/stopped",
        "--worktree",
        str(worktree),
        ok=False,
    )


def test_child_requirement_edit_invalidates_review_even_after_result_adoption(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    g = auto_gate(p, monkeypatch)
    state = p.state()
    state["issues"]["10"]["body"] = state["issues"]["10"]["body"].replace(
        "動作する", "新しい境界を確認する"
    )
    p.save(state)
    p.cli(
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("変更後の要件で成果を確認した。レビューは別途必要。"),
    )
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert rows[100]["decision"] == "blocked"
    assert rows[101]["decision"] == rows[102]["decision"] == "merge"
    assert "issue_10_reviewed_requirements_changed" in rows[100]["reasons"]
    g.review.cli(
        "run",
        "--request",
        str(g.review.request_path),
        "--agent-exec",
        str(g.review.fake),
        ok=False,
    )
    assert g.merge_calls() == [] and len(g.review.calls()) == 1


def test_child_stop_preserves_independent_review_progress_and_merge_coverage(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    g = auto_gate(p, monkeypatch)
    p.cli("stop", "--child", "10", "--spec", p.detail())
    p.cli(
        "review-state",
        "--run-id",
        g.review.request["review_run_id"],
        "--state",
        "complete",
        "--evidence-file",
        p.note("独立PRの証拠を確認。停止子のcoverageはgateで拒否する。"),
    )
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert rows[100]["decision"] == "blocked"
    assert "child_10_stopped" in rows[100]["reasons"]
    assert rows[101]["decision"] == rows[102]["decision"] == "merge"
    assert g.merge_calls() == [] and len(g.review.calls()) == 1


@pytest.mark.parametrize(
    "change", ["version", "missing", "legacy", "generation", "digest"]
)
def test_new_review_manifest_requires_explicit_valid_authorization(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    policy = review_policy_module(monkeypatch)
    fixture = importlib.import_module("tests.e2e.agents_ops.test_parent_review_policy")
    manifest = fixture.request()
    manifest["schema_version"] = 2
    manifest["requirements"] = {"1": "b" * 64}
    manifest["authorization"] = {
        "generation": 1,
        "execution_basis_kind": "autonomous_decision",
        "execution_basis_sha256": "a" * 64,
    }
    if change == "version":
        manifest["schema_version"] = 3
    elif change == "missing":
        manifest.pop("authorization")
    elif change == "legacy":
        manifest["authorization"] = {"generation": 1, "agreement_sha256": "a" * 64}
    elif change == "generation":
        manifest["authorization"]["generation"] = True
    else:
        manifest["authorization"]["execution_basis_sha256"] = "invalid"
    with pytest.raises(ValueError):
        policy.request_manifest(manifest, instructed=False)


def test_registry_migration_preserves_claims_reservations_and_unknown_fields(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("reserved")
    worktree = p.worktree("reserved")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/reserved",
        "--worktree",
        str(worktree),
    )
    path = registry(p)
    old = json.loads(path.read_text())
    old["version"] = 1
    task = old["parents"]["o/r#1"]["tasks"]["10"]
    task.pop("authorization")
    task.pop("requirements_sha256")
    old["parents"]["o/r#1"]["extension"] = {
        "receipts": ["existing"],
        "notes": "preserve",
    }
    path.write_text(json.dumps(old))
    state_mod = importlib.import_module("parent.state")
    with state_mod.Store(path.parent).locked() as store:
        assert store.data["version"] == 1
        assert store.parents == old["parents"]
        assert json.loads(path.read_text())["version"] == 1
        store.save()
    with state_mod.Store(path.parent).locked() as store:
        assert store.parents == old["parents"]
    assert (
        json.loads(p.cli("reconcile").stdout)["tasks"][0]["action"] == "do_not_launch"
    )


@pytest.mark.parametrize(
    "bad", ["truncated", "duplicate", "version", "generation", "task", "review"]
)
def test_malformed_registry_never_becomes_an_empty_assignment(
    parent_env: ParentEnv, bad: str
) -> None:
    p = parent_env
    p.agreed()
    path = registry(p)
    data = json.loads(path.read_text())
    if bad == "truncated":
        path.write_text('{"version":1,"parents":')
    elif bad == "duplicate":
        path.write_text('{"version":1,"parents":{},"parents":{}}')
    else:
        if bad == "version":
            data["version"] = 100
        elif bad == "generation":
            data["parents"]["o/r#1"]["generation"] = True
        elif bad == "task":
            data["parents"]["o/r#1"]["tasks"] = {"10": None}
        else:
            data["parents"]["o/r#1"]["reviews"] = [{"request": {"schema_version": 99}}]
        path.write_text(json.dumps(data))
    before = path.read_bytes()
    calls = len(p.calls())
    p.cli("register", ok=False)
    assert path.read_bytes() == before
    assert len(p.calls()) == calls


def test_interrupted_migration_requires_recovery_without_losing_old_state(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    path = registry(p)
    data = json.loads(path.read_text())
    data["version"] = 1
    path.write_text(json.dumps(data))
    temporary = path.with_name("registry.json.tmp")
    temporary.write_text("interrupted migration")
    before = path.read_bytes()
    p.cli("release", "--note-file", p.note(), ok=False)
    assert path.read_bytes() == before
    assert temporary.read_text() == "interrupted migration"
    temporary.unlink()  # Simulate explicit recovery after inspecting the old state.
    p.cli("release", "--note-file", p.note())
    assert json.loads(path.read_text())["version"] == 1


def test_missing_registry_symlink_target_is_not_an_empty_assignment(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    path = registry(p)
    path.unlink()
    missing = path.with_name("missing-registry.json")
    path.symlink_to(missing)
    p.cli("register", ok=False)
    assert path.is_symlink() and not missing.exists()


@pytest.mark.parametrize("kind", ["parent-session", "parent-review"])
def test_unknown_public_marker_version_is_rejected(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    comment = {
        "user": {"login": "operator"},
        "body": f'<!-- {kind}:v99 {{"parent":1}} -->',
    }
    if kind == "parent-session":
        state = parent_env.state()
        state["comments"]["1"] = [{"id": 9, **comment}]
        parent_env.save(state)
        parent_env.cli("register", ok=False)
    else:
        policy = review_policy_module(monkeypatch)
        with pytest.raises(ValueError, match="unknown parent-review marker version"):
            policy.records([comment], "operator")


def test_legacy_public_review_and_local_result_survive_migration(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    policy = importlib.import_module("parent.policy")
    projection = importlib.import_module("merge_gate_policy").projection
    request = copy.deepcopy(g.review.request)
    request.pop("review_state")
    request.pop("authorization")
    request.pop("requirements")
    request["schema_version"] = 1
    request_sha = policy.digest(request)
    agreement = next(
        e
        for e in policy.events(p.state()["comments"]["1"], "operator", 1)
        if e["event"] == "agreed"
    )
    auth = {"generation": 1, "agreement_sha256": agreement["agreement_sha256"]}
    run_dir = g.review.run_dir
    result = json.loads((run_dir / "result.json").read_text())
    result.update(request=request, request_sha256=request_sha)
    execution = json.loads((run_dir / "execution.json").read_text())
    execution.update(authorization=auth, request_sha256=request_sha)
    for name, data in (
        ("request", request),
        ("result", result),
        ("execution", execution),
    ):
        (run_dir / f"{name}.json").write_text(json.dumps(data))
    state = p.state()
    for rows in state["comments"].values():
        for comment in rows:
            history = (
                policy.events([comment], "operator", 1)
                if comment["body"].startswith(policy.PREFIX)
                else []
            )
            if history and history[0]["event"] == "review":
                event = history[0]
                event.pop("body")
                for key in ("execution_basis_kind", "execution_basis_sha256"):
                    event.pop(key)
                event.update(auth, request_sha256=request_sha)
                comment["body"] = (
                    policy.marker(event)
                    + "\n\n```json\n"
                    + json.dumps(request)
                    + "\n```"
                )
            elif history and history[0]["event"] == "agreed":
                event = history[0]
                event.pop("body")
                event.pop("comment_inputs")
                comment["body"] = policy.marker(event)
            records = g.policy.records([comment], "operator")
            if records and records[0]["kind"] == "result":
                comment["body"] = g.policy.marker(result)
            elif records and records[0]["kind"] == "pr":
                comment["body"] = g.policy.marker(
                    projection(result, records[0]["pr"], execution["parent_run_url"])
                )
    p.save(state)
    old = json.loads(registry(p).read_text())
    old["version"] = 1
    old["parents"]["o/r#1"]["reviews"][-1]["request"] = request
    registry(p).write_text(json.dumps(old))
    result_bytes = (run_dir / "result.json").read_bytes()
    response = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert response.returncode == 0, response.stderr + response.stdout
    assert all(row["decision"] == "merge" for row in g.results(response).values())
    p.cli(
        "review-state",
        "--run-id",
        request["review_run_id"],
        "--state",
        "complete",
        "--evidence-file",
        p.note(),
    )
    assert json.loads(registry(p).read_text())["version"] == 1
    assert (run_dir / "result.json").read_bytes() == result_bytes
    assert len(g.review.calls()) == 1


def ordinary_comment(p: ParentEnv, body: str, login: str = "human") -> dict[str, Any]:
    state = p.state()
    comment = {
        "id": 1 + max(c["id"] for rows in state["comments"].values() for c in rows),
        "user": {"login": login},
        "body": body,
    }
    state["comments"]["1"].append(comment)
    p.save(state)
    return comment


def ignored_comment(comment: dict[str, Any]) -> dict[str, Any]:
    execution = importlib.import_module("parent.execution")
    return {
        "id": comment["id"],
        "sha256": execution.digest(execution.comment_snapshot(comment)),
        "reason": "要件と関係しない雑談であることを本文から確認した。",
    }


@pytest.mark.parametrize("change", ["added", "edited", "deleted"])
def test_human_agreement_comment_changes_block_delegation_review_and_merge(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    comment = ordinary_comment(
        p, "作業を停止してください。" if change == "added" else "要件と関係しない雑談。"
    )
    if change != "added":
        p.cli(
            "agreement-ignore-comments",
            "--ignored-comments",
            p.spec([ignored_comment(comment)]),
        )
        state = p.state()
        if change == "edited":
            next(c for c in state["comments"]["1"] if c["id"] == comment["id"])[
                "body"
            ] = "作業を停止してください。"
        else:
            state["comments"]["1"] = [
                c for c in state["comments"]["1"] if c["id"] != comment["id"]
            ]
        p.save(state)
    worktree = p.worktree("unclassified")
    before = p.state(), registry(p).read_bytes()
    responses = [
        p.child("unclassified", ok=False),
        p.cli(
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/unclassified",
            "--worktree",
            str(worktree),
            ok=False,
        ),
        p.cli(
            "review-request",
            "--run-id",
            g.review.request["review_run_id"],
            "--instruction-file",
            p.note(),
            ok=False,
        ),
        g.cli("merge"),
    ]
    for response in responses:
        assert response.returncode == 2
        error = (
            json.loads(response.stderr)["error"]
            if response.stderr.startswith("{")
            else response.stderr
        )
        assert f"ID={comment['id']}" in error
        assert f"https://github.com/o/r/issues/1#issuecomment-{comment['id']}" in error
        assert f"({change})" in error
        assert "agreement-ignore-comments" in error and "checking → agree" in error
    p.cli("status")  # Stale authorization never prevents reading an existing parent.
    assert (p.state(), registry(p).read_bytes()) == before
    assert g.merge_calls() == [] and len(g.review.calls()) == 1


@pytest.mark.parametrize("gate_env", ["shared"], indirect=True)
def test_human_unrelated_comment_classification_preserves_review(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    frozen = (g.review.run_dir / "result.json").read_bytes(), registry(p).read_bytes()
    comments = [
        ordinary_comment(p, "要件と無関係な雑談。", login)
        for login in ("operator", "human")
    ]
    entries = [ignored_comment(c) for c in comments]
    for _ in range(2):
        p.cli("agreement-ignore-comments", "--ignored-comments", p.spec(entries))
    assert (
        sum(
            '"event": "agreement_comments"' in c["body"]
            for c in p.state()["comments"]["1"]
        )
        == 1
    )
    assert (
        (g.review.run_dir / "result.json").read_bytes(),
        registry(p).read_bytes(),
    ) == frozen
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert all(row["decision"] == "merge" for row in rows.values())
    assert "shared_exemption" not in rows[100]
    assert g.merge_calls() == [] and len(g.review.calls()) == 1


@pytest.mark.parametrize(
    "change", ["digest", "duplicate", "missing", "shape", "partial"]
)
def test_human_comment_classification_rejects_invalid_or_incomplete_input(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.agreed()
    comment = ordinary_comment(p, "雑談。")
    entries = [ignored_comment(comment)]
    if change == "digest":
        entries[0]["sha256"] = "a" * 64
    elif change == "duplicate":
        entries *= 2
    elif change == "missing":
        entries[0]["id"] += 100
    elif change == "shape":
        entries[0]["unexpected"] = True
    else:
        ordinary_comment(p, "別の未分類指示。")
    before = p.state(), registry(p).read_bytes()
    p.cli("agreement-ignore-comments", "--ignored-comments", p.spec(entries), ok=False)
    assert (p.state(), registry(p).read_bytes()) == before


def test_human_comment_reclassification_records_return_to_previous_text(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    comment = ordinary_comment(p, "要件と無関係な雑談A。")
    for body in (
        "要件と無関係な雑談A。",
        "要件と無関係な雑談B。",
        "要件と無関係な雑談A。",
    ):
        state = p.state()
        comment = next(c for c in state["comments"]["1"] if c["id"] == comment["id"])
        comment["body"] = body
        p.save(state)
        for _ in range(2):
            p.cli(
                "agreement-ignore-comments",
                "--ignored-comments",
                p.spec([ignored_comment(comment)]),
            )
    policy = importlib.import_module("parent.policy")
    records = [
        e
        for e in policy.events(p.state()["comments"]["1"], "operator", 1)
        if e["event"] == "agreement_comments"
    ]
    assert [e["classification_revision"] for e in records] == [1, 2, 3]
    assert records[0]["comment_inputs"] == records[2]["comment_inputs"]
    p.child("reclassified")


@pytest.mark.parametrize("change", ["added", "edited"])
def test_legacy_human_agreement_remains_readable_and_requires_reagreement_for_changes(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.cli("register")
    comment = ordinary_comment(p, "合意前の指示。")
    p.agreed()
    policy = importlib.import_module("parent.policy")
    state = p.state()
    agreed = next(c for c in state["comments"]["1"] if '"event": "agreed"' in c["body"])
    [event] = policy.events([agreed], "operator", 1)
    agreement = event["agreement"]
    event.pop("body")
    event.pop("comment_inputs")
    agreed["body"] = policy.marker(event)
    agreed["created_at"] = "2026-10-22T03:00:00Z"
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    p.cli("status")
    assert (p.state(), registry(p).read_bytes()) == before
    p.child("legacy-compatible")
    if change == "added":
        comment = ordinary_comment(p, "作業を止めてください。")
    else:
        state = p.state()
        original = next(c for c in state["comments"]["1"] if c["id"] == comment["id"])
        original.update(
            body="対象範囲を変更します。", updated_at="2026-10-22T03:01:00Z"
        )
        p.save(state)
    response = p.child("blocked", ok=False)
    assert f"ID={comment['id']}" in response.stderr and f"({change})" in response.stderr
    p.cli("status")
    p.cli(
        "agree", "--spec", p.spec(agreement), ok=False
    )  # Retrying agree is not consent.
    p.cli("checking")
    p.cli("agree", "--spec", p.spec(agreement))
    p.child("reagreed")


@pytest.mark.parametrize("change", ["edited", "deleted"])
def test_human_agreement_detects_changes_to_existing_instruction_comments(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.cli("register")
    comment = ordinary_comment(p, "合意に含まれた追加要件。")
    p.agreed()
    state = p.state()
    if change == "edited":
        next(c for c in state["comments"]["1"] if c["id"] == comment["id"])["body"] = (
            "要件を変更する。"
        )
    else:
        state["comments"]["1"] = [
            c for c in state["comments"]["1"] if c["id"] != comment["id"]
        ]
    p.save(state)
    response = p.child("stale-comment", ok=False)
    assert f"ID={comment['id']}" in response.stderr and f"({change})" in response.stderr


@pytest.mark.parametrize("change", ["replace", "add"])
def test_changed_closing_targets_require_new_review_before_instruction(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.agreed()
    p.child("original")
    p.child("new-target")
    state = p.state()
    state["prs"] = [p.pr(100, 10)]
    p.save(state)
    frozen = json.loads(p.cli("review-prepare", "--prs", "100").stdout)
    state = p.state()
    state["prs"][0]["body"] = (
        "Closes #11" if change == "replace" else "Closes #10\nCloses #11"
    )
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    response = p.cli(
        "review-request",
        "--run-id",
        frozen["review_run_id"],
        "--instruction-file",
        p.note(),
        ok=False,
    )
    assert "closing targets differ from frozen requirements" in response.stderr
    assert (p.state(), registry(p).read_bytes()) == before
    new = json.loads(
        p.cli(
            "review-prepare",
            "--prs",
            "100",
            "--reason-file",
            p.note("closing対象の変更を確認したため新しいレビューを要求する。"),
        ).stdout
    )
    assert (
        "11" in new["requirements"] and new["review_run_id"] != frozen["review_run_id"]
    )
    p.cli(
        "review-request",
        "--run-id",
        new["review_run_id"],
        "--instruction-file",
        p.note(),
    )
    state = p.state()
    state["issues"]["11"]["body"] = state["issues"]["11"]["body"].replace(
        "動作する", "追加の受入条件を満たす"
    )
    p.save(state)
    p.cli("record-result", "--child", "11", "--evidence-file", p.note())
    response = p.cli(
        "review-request",
        "--run-id",
        new["review_run_id"],
        "--instruction-file",
        p.note(),
        ok=False,
    )
    assert "#11: reviewed requirements changed" in response.stderr


def test_review_collection_rejects_new_closing_child_after_result_adoption(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    r = make_review_env(p)
    p.child("new-closing-target")
    state = p.state()
    state["prs"][0]["body"] = "Closes #13"
    state["issues"]["13"]["body"] = state["issues"]["13"]["body"].replace(
        "動作する", "変更された要件で動作する"
    )
    p.save(state)
    p.cli("record-result", "--child", "13", "--evidence-file", p.note())
    result = json.loads(r.run(ok=False).stdout)["result"]
    assert result["status"] == "failed"
    assert "closing targets differ from frozen requirements" in result["error"]
    assert r.calls() == []


@pytest.mark.parametrize("gate_env", ["multiple"], indirect=True)
def test_result_and_merge_gate_reject_missing_closing_child_requirements(
    gate_env: GateEnv,
) -> None:
    g = gate_env
    result_file = g.review.run_dir / "result.json"
    result = json.loads(result_file.read_text())
    result["request"]["requirements"].pop("11")
    result["request_sha256"] = g.policy.digest(result["request"])
    with pytest.raises(
        ValueError, match="closing targets differ from frozen requirements"
    ):
        g.policy.validate_result(result)
    # Persist a record accepted by the old reader to exercise the real gate too.
    result_file.write_text(json.dumps(result))
    response = g.cli("merge")
    assert response.returncode == 2
    assert "closing targets differ from frozen requirements" in response.stderr
    assert g.merge_calls() == [] and len(g.review.calls()) == 1


@pytest.mark.parametrize("stage", ["prepare", "collect"])
def test_closing_target_changes_during_snapshot_collection_are_rejected(
    parent_env: ParentEnv, stage: str
) -> None:
    p = parent_env
    r = make_review_env(p)
    p.child("new-target")
    p.env["PARENT_CHANGE_CLOSING_ON_READ"] = "Closes #13"
    if stage == "prepare":
        before = registry(p).read_bytes()
        response = p.cli(
            "review-prepare",
            "--prs",
            "100",
            "101",
            "102",
            "--reason-file",
            p.note(),
            ok=False,
        )
        assert "PR changed while freezing" in response.stderr
        assert registry(p).read_bytes() == before
    else:
        result = json.loads(r.run(ok=False).stdout)["result"]
        assert result["status"] == "failed"
        assert "PR set/snapshots changed during collection" in result["error"]
    assert r.calls() == []
