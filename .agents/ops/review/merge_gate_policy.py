"""Parent-run evidence and per-PR merge decisions. No I/O or agent execution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from issues import Issue, task_status
from parent.execution import authorization_revoked, task_errors
from parent.policy import (
    closing_issues,
    dependencies,
    digest,
    progress,
    review_state,
)
from parent_review_policy import (
    BaseAdvance,
    evaluate_coverage,
    pr_findings,
    records,
    validate_result,
)
from review_policy import (
    AuthorConflictError,
    Check,
    GateDecision,
    GateInput,
    ReviewOutcome,
    detect_author_agent,
    evaluate_gate,
    normalize_rollup,
)

REQUIRED_CHECK = "Python"


def python_ci_errors(runs: Sequence[Mapping[str, Any]], head: str) -> tuple[str, ...]:
    """Require actual success from every latest Python check run at this HEAD."""
    required = [run for run in runs if run.get("name") == REQUIRED_CHECK]
    if not required:
        return ("ci_python_missing",)
    reasons = []
    for run in required:
        if not isinstance(run.get("head_sha"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", run["head_sha"]
        ):
            raise ValueError("Python check run has no valid head_sha")
        if run["head_sha"] != head:
            reasons.append("ci_python_head_mismatch")
        status = run.get("status")
        if status not in {
            "queued",
            "in_progress",
            "completed",
            "requested",
            "waiting",
            "pending",
        }:
            raise ValueError(f"unknown Python check run status: {status!r}")
        if status != "completed":
            reasons.append("ci_python_pending")
            continue
        conclusion = run.get("conclusion")
        if conclusion not in {
            "success",
            "failure",
            "neutral",
            "cancelled",
            "skipped",
            "timed_out",
            "action_required",
            "startup_failure",
            "stale",
        }:
            raise ValueError(f"unknown Python check run conclusion: {conclusion!r}")
        if conclusion != "success":
            reasons.append(f"ci_python_{conclusion}")
    return tuple(dict.fromkeys(reasons))


def auto_merge(value: str | None) -> bool:
    if value not in {None, "0", "1"}:
        raise ValueError("AGENT_AUTO_MERGE must be unset, 0 or 1")
    return value == "1"


def projection(result: Mapping[str, Any], n: int, url: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pr",
        "review_run_id": result["review_run_id"],
        "parent_issue": result["request"]["parent_issue"],
        "pr": n,
        "request_sha256": result["request_sha256"],
        "result_sha256": digest(result),
        "snapshot": next(p for p in result["request"]["prs"] if p["number"] == n),
        "coverage": next(p for p in result["prs"] if p["number"] == n)["coverage"],
        "findings": pr_findings(result, n),
        "parent_run_url": url,
    }


def publication_errors(
    result: Mapping[str, Any],
    execution: Mapping[str, Any],
    parent_comments: Sequence[Mapping[str, Any]],
    pr_comments: Mapping[int, Sequence[Mapping[str, Any]]],
    login: str,
    authorization: Mapping[str, Any],
) -> tuple[str, ...]:
    """Require the local committed result and every public projection to agree."""
    validate_result(dict(result))
    errors = []
    if (
        execution.get("status") != "published"
        or execution.get("request_sha256") != result["request_sha256"]
        or execution.get("authorization") != authorization
        or type(execution.get("agent_invocations")) is not int
        or execution["agent_invocations"] != 1
    ):
        errors.append("review_not_published_or_authorized")
    run = result["review_run_id"]
    parent_records = [
        r for r in records(parent_comments, login) if r.get("review_run_id") == run
    ]
    outcomes = [r for r in parent_records if r["kind"] == "result"]
    if (
        not outcomes
        or any(r != result for r in outcomes)
        or any(r["kind"] == "pending" for r in parent_records)
    ):
        errors.append("parent_result_missing_or_inconsistent")
    url = execution.get("parent_run_url")
    urls = {
        f"https://github.com/{result['request']['repo']}/issues/{result['request']['parent_issue']}#issuecomment-{c['id']}"
        for c in parent_comments
        if c.get("id") and records([c], login) == [result]
    }
    if url not in urls:
        errors.append("parent_result_url_mismatch")
    for pr in result["prs"]:
        n = pr["number"]
        found = [
            r
            for r in records(pr_comments.get(n, []), login)
            if r.get("review_run_id") == run and r["kind"] == "pr"
        ]
        if not found or any(r != projection(result, n, str(url)) for r in found):
            errors.append(f"pr_{n}_projection_missing_or_inconsistent")
    return tuple(errors)


def confirmed_state(
    history: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    auth: Mapping[str, Any],
) -> str | None:
    """Preserve this run's authorization unless its agreement or owner is revoked."""
    revoked_at = max(
        (i for i, e in enumerate(history) if authorization_revoked(e, auth)), default=-1
    )
    recent = history[revoked_at + 1 :]
    relevant = [
        e
        for e in recent
        if e.get("event") == "review"
        and e.get("review_run_id") == request["review_run_id"]
    ]
    if not relevant or any(
        any(e.get(k) != v for k, v in auth.items()) for e in relevant
    ):
        return None
    return review_state(recent, request)


def dependency_order(
    prs: Sequence[Mapping[str, Any]], children: Mapping[int, Issue], repo: str
) -> list[int]:
    """Stable dependency order; cycles and dependencies outside this parent are errors."""
    progress(children, {})
    linked = {
        p["number"]: closing_issues(p.get("body") or "", repo) & set(children)
        for p in prs
    }
    result: list[int] = []
    visiting: set[int] = set()

    def visit(n: int) -> None:
        if n in visiting:
            raise ValueError("cyclic PR dependencies")
        if n in result:
            return
        visiting.add(n)
        deps = {
            d for child in linked[n] for d in dependencies(children[child].body)
        } - linked[n]
        for other in sorted(linked):
            if other != n and linked[other] & deps:
                visit(other)
        visiting.remove(n)
        result.append(n)

    for n in sorted(linked):
        visit(n)
    return result


def dependency_status(
    child: Issue,
    pulls: Sequence[Mapping[str, Any]],
    repo: str,
    default_branch: str,
) -> str:
    """Distinguish a completed investigation from unfinished or misplaced code."""
    if task_status(child) != "completed":
        return "not_completed"
    related = [
        p for p in pulls if child.number in closing_issues(p.get("body") or "", repo)
    ]
    if not related:
        return "completed_without_pr"
    if any(
        p.get("state") not in {"open", "closed"}
        or "merged_at" not in p
        or (
            p["merged_at"] is not None
            and (not isinstance(p["merged_at"], str) or not p["merged_at"].strip())
        )
        for p in related
    ):
        return "pr_state_unknown"
    if any(p["state"] == "open" for p in related):
        return "open_pr"
    merged = [p for p in related if p["merged_at"] is not None]
    if not merged:
        return "unmerged"
    if any(
        not isinstance(p.get("base"), Mapping)
        or not isinstance(p["base"].get("ref"), str)
        or not p["base"]["ref"].strip()
        for p in merged
    ):
        return "pr_base_unknown"
    if any(p["base"]["ref"] == default_branch for p in merged):
        return "merged_into_default_branch"
    return "base_not_default_branch"


@dataclass(frozen=True)
class MergeInput:
    repo: str
    default_branch: str
    parent_issue: int
    pr: Mapping[str, Any]
    children: Mapping[int, Issue]
    tasks: Mapping[str, Any]
    pulls: Sequence[Mapping[str, Any]]
    result: Mapping[str, Any] | None
    current: Sequence[Mapping[str, Any]]
    resolutions: Sequence[Mapping[str, Any]]
    fix_paths: Mapping[int, Sequence[str]]
    fix_diff_digests: Mapping[int, str]
    changed_paths: tuple[str, ...]
    review_state: str | None
    evidence_errors: tuple[str, ...]
    base_advances: Sequence[BaseAdvance] = ()
    inherited_snapshots: frozenset[tuple[str, str]] = frozenset()
    parent_needs_human: bool = False


@dataclass(frozen=True)
class MergeResult:
    decision: GateDecision
    reasons: tuple[str, ...]
    dependency_statuses: Mapping[int, str] = field(default_factory=dict)
    ci_checks: tuple[Check, ...] | None = None
    ci_head_sha: str | None = None


def evaluate_parent_gate(gate: MergeInput) -> MergeResult:
    pr, n = gate.pr, gate.pr["number"]
    reasons = list(gate.evidence_errors)
    labels = {label["name"] for label in pr["labels"]}
    needs_human = gate.parent_needs_human or "needs-human" in labels
    if pr["base"]["ref"] != gate.default_branch:
        reasons.append("base_not_default_branch")
    if pr["state"] != "open" or pr.get("merged_at") is not None:
        reasons.append("pr_not_open")
    if type(pr.get("draft")) is not bool:
        reasons.append("draft_unknown")
    try:
        if detect_author_agent(pr["head"]["ref"], labels) is None:
            reasons.append("author_unknown")
    except AuthorConflictError:
        reasons.append("author_conflict")
    targets = set(gate.children) or {gate.parent_issue}
    linked = closing_issues(pr.get("body") or "", gate.repo) & targets
    if not linked:
        reasons.append("not_direct_child_pr")
    states = progress(gate.children, gate.tasks)
    dependency_statuses = {}
    for child in sorted(linked & set(gate.children)):
        needs_human |= "needs-human" in gate.children[child].labels
        if states[child] != "continue":
            reasons.append(f"child_{child}_{states[child]}")
        branch = gate.tasks.get(str(child), {}).get("branch")
        task = gate.tasks.get(str(child), {})
        auth = gate.result["request"].get("authorization") if gate.result else None
        if auth:
            reasons.extend(
                f"child_{child}_{error}"
                for error in task_errors(task, auth, gate.children[child].body)
            )
        if branch and branch != pr["head"]["ref"]:
            reasons.append(f"child_{child}_branch_mismatch")
        duplicates = [
            p
            for p in gate.pulls
            if p["state"] == "open"
            and child in closing_issues(p.get("body") or "", gate.repo)
        ]
        if len(duplicates) != 1:
            reasons.append(f"child_{child}_ambiguous_pr")
        for dep in dependencies(gate.children[child].body):
            if dep in linked:
                continue
            status = dependency_status(
                gate.children[dep], gate.pulls, gate.repo, gate.default_branch
            )
            dependency_statuses[dep] = status
            if status not in {"completed_without_pr", "merged_into_default_branch"}:
                reasons.append(f"dependency_{dep}_{status}")
    review = None
    if gate.result is not None:
        result = validate_result(dict(gate.result))
        if (
            result["request"]["parent_issue"] != gate.parent_issue
            or result["request"]["repo"].casefold() != gate.repo.casefold()
        ):
            reasons.append("review_parent_mismatch")
        entry = next((p for p in result["prs"] if p["number"] == n), None)
        if entry is None:
            reasons.append("pr_outside_review_targets")
        else:
            if entry["child_issues"] != sorted(linked):
                reasons.append("review_child_assignment_changed")
            coverage = evaluate_coverage(
                result,
                gate.current,
                gate.resolutions,
                gate.fix_paths,
                fix_diff_digests=gate.fix_diff_digests,
                base_advances=gate.base_advances,
                inherited_snapshots=gate.inherited_snapshots,
            )
            # Global identity/set errors block all; another PR's coverage/finding does
            # not delay an independently covered PR. Merged targets remain in current.
            reasons.extend(
                r
                for r in coverage.reasons
                if not re.match(r"pr_[1-9][0-9]*_", r) or r.startswith(f"pr_{n}_")
            )
            unresolved = [f for p, f in coverage.unresolved if p == n]
            if unresolved:
                reasons.append("unresolved_findings")
            if not reasons and not unresolved:
                review = ReviewOutcome("approve", 0)
    if gate.review_state not in {"fixes_verified", "complete"}:
        reasons.append("parent_review_not_confirmed")
    if pr["state"] == "open":
        reasons.extend(python_ci_errors(pr["ci_check_runs"], pr["head"]["sha"]))
    mergeable = pr.get("mergeable")
    if mergeable is not None and type(mergeable) is not bool:
        raise ValueError("REST mergeable must be true, false or null")
    base = evaluate_gate(
        GateInput(
            is_draft=pr.get("draft") is not False,
            is_cross_repository=pr["head"]["repo"]["full_name"].casefold()
            != gate.repo.casefold(),
            mergeable="MERGEABLE"
            if mergeable is True
            else "CONFLICTING"
            if mergeable is False
            else "UNKNOWN",
            review=review,
        )
    )
    reasons.extend(base.reasons)
    if needs_human:
        reasons.append("human_judgment_pending")
    reasons = list(dict.fromkeys(reasons))
    human_reasons = {"human_judgment_pending"}
    decision = (
        GateDecision.BLOCKED
        if set(reasons) - human_reasons
        else GateDecision.NEEDS_HUMAN
        if reasons
        else GateDecision.MERGE
    )
    return MergeResult(
        decision,
        tuple(reasons),
        dependency_statuses,
        ci_checks=normalize_rollup(pr["statusCheckRollup"])
        if pr["state"] == "open"
        else None,
        ci_head_sha=pr["ci_head_sha"],
    )
