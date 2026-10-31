"""Pure merge decisions and publication checks, without Git/GitHub/agent setup."""

from __future__ import annotations

import copy
import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

HEAD = "a" * 40
CI_ERRORS = {
    "success": (),
    "failure": ("ci_python_failure",),
    "pending": ("ci_python_pending",),
    "cancelled": ("ci_python_cancelled",),
    "missing": ("ci_python_missing",),
    "head_mismatch": ("ci_python_head_mismatch",),
}


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    ops = Path(__file__).resolve().parents[3] / ".agents/ops"
    monkeypatch.syspath_prepend(str(ops))
    monkeypatch.syspath_prepend(str(ops / "review"))
    return importlib.import_module("merge_gate_policy")


def ci_runs(state: str, head: str = HEAD) -> list[dict[str, Any]]:
    if state == "missing":
        return []
    status, conclusion = {
        "success": ("completed", "success"),
        "failure": ("completed", "failure"),
        "pending": ("queued", None),
        "cancelled": ("completed", "cancelled"),
        "head_mismatch": ("completed", "success"),
    }[state]
    return [
        {
            "name": "Python",
            "head_sha": "0" * 40 if state == "head_mismatch" else head,
            "status": status,
            "conclusion": conclusion,
            "started_at": "2026-10-30T01:00:00Z",
        }
    ]


def valid_gate(policy: Any, *, dependency: bool = False) -> Any:
    """Literal current-schema evidence; the policy validates it on every call."""
    child = policy.Issue(
        10,
        "Child",
        "## 依存関係\n#11 の完了後。" if dependency else "## 依存関係\nなし",
        frozenset(),
        parent=1,
    )
    children = {10: child}
    if dependency:
        children[11] = policy.Issue(
            11, "Dependency", "## 依存関係\nなし", frozenset(), parent=1
        )
    auth = {
        "generation": 1,
        "execution_basis_kind": "human_agreement",
        "execution_basis_sha256": "e" * 64,
    }
    snapshot = {
        "number": 100,
        "head_sha": HEAD,
        "base_sha": "b" * 40,
        "base_ref": "main",
        "merge_base_sha": "b" * 40,
        "diff_sha256": "c" * 64,
        "branch": "codex/unit",
    }
    request = {
        "schema_version": 2,
        "review_run_id": "parent-1-" + "d" * 32,
        "repo": "o/r",
        "parent_issue": 1,
        "reason": "Review this change.",
        "prs": [copy.deepcopy(snapshot)],
        "authorization": auth,
        "requirements": {"1": "f" * 64, "10": policy.digest(child.body)},
    }
    result = {
        "schema_version": 1,
        "kind": "result",
        "review_run_id": request["review_run_id"],
        "request_sha256": policy.digest(request),
        "request": request,
        "reviewer": "codex",
        "profile": "review",
        "started_at": "2026-10-30T00:00:00+00:00",
        "finished_at": "2026-10-30T00:01:00+00:00",
        "status": "succeeded",
        "verdict": "approve",
        "summary": "Reviewed.",
        "error": None,
        "exception": None,
        "cross_findings": [],
        "prs": [
            {
                "number": 100,
                "child_issues": [10],
                "coverage": "reviewed",
                "summary": "Reviewed.",
                "findings": [],
            }
        ],
    }
    pr = {
        "number": 100,
        "body": "Closes #10",
        "labels": [],
        "state": "open",
        "merged_at": None,
        "draft": False,
        "mergeable": True,
        "head": {"sha": HEAD, "ref": "codex/unit", "repo": {"full_name": "o/r"}},
        "base": {"ref": "main"},
        "ci_check_runs": ci_runs("success"),
        "ci_head_sha": HEAD,
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": "Python",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            }
        ],
    }
    return policy.MergeInput(
        repo="o/r",
        default_branch="main",
        parent_issue=1,
        pr=pr,
        children=children,
        tasks={
            "10": {
                "branch": "codex/unit",
                "delegated": True,
                "authorization": auth,
                "requirements_sha256": policy.digest(child.body),
            }
        },
        pulls=[copy.deepcopy(pr)],
        result=result,
        current=[snapshot],
        resolutions=[],
        fix_paths={},
        fix_diff_digests={},
        changed_paths=("change.txt",),
        review_state="fixes_verified",
        evidence_errors=(),
    )


def refuse(policy: Any, gate: Any, case: str) -> Any:
    pr, result = gate.pr, gate.result
    if case == "no_review":
        # File absence is collected by the CLI; the policy must preserve that
        # evidence error even when all other inputs and Python CI permit merging.
        return replace(
            gate, result=None, evidence_errors=("parent_review_missing_or_interrupted",)
        )
    if case in {"not_received", "fixing"}:
        gate = replace(gate, review_state=None if case == "not_received" else "fixing")
    elif case == "outside":
        extra = {**pr, "number": 103}
        gate = replace(
            gate,
            pr=extra,
            pulls=[*gate.pulls, extra],
            current=[*gate.current, {**gate.current[0], "number": 103}],
        )
    elif case == "not_reviewed":
        result.update(status="failed", verdict="incomplete", error="Not reviewed.")
        result["prs"][0]["coverage"] = "not_reviewed"
    elif case in {"major", "blocker"}:
        result["verdict"] = "request_changes"
        result["prs"][0]["findings"] = [
            {
                "id": "pr-100-1",
                "severity": case,
                "path": "change.txt",
                "line": 1,
                "rule": "Correctness",
                "detail": "Requires a fix.",
            }
        ]
    elif case == "head":
        pr["head"]["sha"] = "f" * 40
        gate.current[0]["head_sha"] = "f" * 40
    elif case == "base_ref":
        pr["base"]["ref"] = "other"
    elif case == "draft":
        pr["draft"] = True
    elif case == "fork":
        pr["head"]["repo"]["full_name"] = "fork/r"
    elif case == "closed":
        pr["state"] = "closed"
    elif case == "author":
        pr["head"]["ref"] = "unknown/source"
    elif case in {"conflict", "unknown"}:
        pr["mergeable"] = False if case == "conflict" else None
    elif case == "human":
        pr["labels"] = [{"name": "needs-human"}]
    elif case == "parent_human":
        gate = replace(gate, parent_needs_human=True)
    elif case == "child_human":
        gate.children[10] = replace(
            gate.children[10], labels=frozenset({"needs-human"})
        )
    elif case == "stopped":
        gate.tasks["10"]["stopped"] = True
    elif case not in {
        "interrupted",
        "parent_missing",
        "projection_missing",
        "projection_digest",
        "failed_record",
        "foreign_result",
        "quoted_result",
    }:
        raise AssertionError(f"Unknown refusal case: {case}")

    public = importlib.import_module("parent_review_policy")
    auth = result["request"]["authorization"]
    url = "https://github.com/o/r/issues/1#issuecomment-7"
    execution = {
        "status": "published",
        "request_sha256": result["request_sha256"],
        "authorization": auth,
        "agent_invocations": 1,
        "parent_run_url": url,
    }
    parent_comments = [
        {"id": 7, "user": {"login": "operator"}, "body": public.marker(result)}
    ]
    projection = policy.projection(result, 100, url)
    pr_comments = {
        100: [
            {"id": 8, "user": {"login": "operator"}, "body": public.marker(projection)}
        ]
    }
    if case == "interrupted":
        execution["status"] = "running"
    elif case == "parent_missing":
        parent_comments[0]["body"] = "<!-- agent-cross-review:v4 approve -->"
    elif case == "foreign_result":
        parent_comments[0]["user"]["login"] = "attacker"
    elif case == "quoted_result":
        parent_comments[0]["body"] = "Quoted\n" + parent_comments[0]["body"]
    elif case == "failed_record":
        failed = copy.deepcopy(result)
        failed.update(
            status="failed", verdict="incomplete", error="Publication failed."
        )
        failed["prs"][0]["coverage"] = "not_reviewed"
        parent_comments.append(
            {"id": 9, "user": {"login": "operator"}, "body": public.marker(failed)}
        )
    elif case == "projection_missing":
        pr_comments[100] = []
    elif case == "projection_digest":
        projection["result_sha256"] = "f" * 64
        pr_comments[100][0]["body"] = public.marker(projection)
    return replace(
        gate,
        evidence_errors=policy.publication_errors(
            result, execution, parent_comments, pr_comments, "operator", auth
        ),
    )


@pytest.mark.parametrize("state,errors", list(CI_ERRORS.items()))
def test_python_ci_errors(policy: Any, state: str, errors: tuple[str, ...]) -> None:
    assert policy.python_ci_errors(ci_runs(state), HEAD) == errors


@pytest.mark.parametrize("state", list(CI_ERRORS))
def test_ci_alone_controls_merge(policy: Any, state: str) -> None:
    gate = valid_gate(policy)
    gate.pr["ci_check_runs"] = ci_runs(state)
    outcome = policy.evaluate_parent_gate(gate)
    assert outcome.reasons == CI_ERRORS[state]
    assert outcome.decision.value == ("merge" if state == "success" else "blocked")


@pytest.mark.parametrize(
    "case,reason",
    [
        pytest.param(case, reason, id=case)
        for case, reason in [
            ("no_review", "parent_review_missing_or_interrupted"),
            ("not_received", "parent_review_not_confirmed"),
            ("fixing", "parent_review_not_confirmed"),
            ("interrupted", "review_not_published_or_authorized"),
            ("outside", "pr_outside_review_targets"),
            ("not_reviewed", "pr_100_not_reviewed"),
            ("major", "unresolved_findings"),
            ("blocker", "unresolved_findings"),
            ("head", "pr_100_unverified_head"),
            ("base_ref", "base_not_default_branch"),
            ("draft", "draft"),
            ("fork", "fork"),
            ("closed", "pr_not_open"),
            ("author", "author_unknown"),
            ("conflict", "conflict"),
            ("unknown", "mergeable_unknown"),
            ("human", "human_judgment_pending"),
            ("parent_human", "human_judgment_pending"),
            ("child_human", "human_judgment_pending"),
            ("stopped", "child_10_stopped"),
            ("parent_missing", "parent_result_missing_or_inconsistent"),
            ("projection_missing", "pr_100_projection_missing_or_inconsistent"),
            ("projection_digest", "pr_100_projection_missing_or_inconsistent"),
            ("failed_record", "parent_result_missing_or_inconsistent"),
            ("foreign_result", "parent_result_missing_or_inconsistent"),
            ("quoted_result", "parent_result_missing_or_inconsistent"),
        ]
    ],
)
@pytest.mark.parametrize("ci_state", list(CI_ERRORS))
def test_refusals_survive_each_ci_state(
    policy: Any, case: str, reason: str, ci_state: str
) -> None:
    gate = refuse(policy, valid_gate(policy), case)
    gate.pr["ci_check_runs"] = ci_runs(ci_state, gate.pr["head"]["sha"])
    outcome = policy.evaluate_parent_gate(gate)
    assert reason in outcome.reasons
    human_only = (
        case in {"human", "parent_human", "child_human"} and ci_state == "success"
    )
    assert outcome.decision.value == ("needs_human" if human_only else "blocked")
    # Closed PRs do not query CI; open PRs keep CI reasons alongside the refusal.
    assert {r for r in outcome.reasons if r.startswith("ci_python_")} == (
        set() if case == "closed" else set(CI_ERRORS[ci_state])
    )


@pytest.mark.parametrize("ci_state", list(CI_ERRORS))
def test_dependency_survives_each_ci_state(policy: Any, ci_state: str) -> None:
    gate = valid_gate(policy, dependency=True)
    gate.pr["ci_check_runs"] = ci_runs(ci_state)
    outcome = policy.evaluate_parent_gate(gate)
    assert outcome.decision.value == "blocked"
    assert "dependency_11_not_completed" in outcome.reasons
    assert outcome.dependency_statuses == {11: "not_completed"}
