"""Resolve execution targets from formal relations, independently of role labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from issues import Issue
from issues.evidence import require_work_issue

from .policy import closing_issues, dependencies, replace_section


class WorkTargetsChanged(ValueError):
    """Existing work would be orphaned by a change to the formal target set."""


def worker_ended(task: Mapping[str, Any]) -> bool:
    if not task.get("delegated"):
        return True
    proof = task.get("worker_ended")
    return bool(
        isinstance(proof, dict)
        and proof.get("delegation_id") == task.get("delegation_id", "legacy")
        and isinstance(proof.get("evidence"), str)
        and proof["evidence"].strip()
    )


def stopped_self_work(task: Mapping[str, Any]) -> bool:
    """An explicit stop plus termination can retire work with no accepted result.

    Results/PRs require explicit ownership reconciliation instead of automatic
    transfer. The local worker history is retained even when no work was done.
    """
    return bool(
        task.get("stopped") and not task.get("verification") and worker_ended(task)
    )


def resolve(
    parent: Issue,
    children: dict[int, Issue],
    tasks: Mapping[str, Any],
    *,
    managed: bool,
    reviewed_targets: set[int] | None = None,
) -> dict[int, Issue]:
    for item in [parent, *children.values()]:
        require_work_issue({"labels": item.labels, "body": item.body})
    if set(children) != set(parent.children) or any(
        n == parent.number or child.number != n or child.parent != parent.number
        for n, child in children.items()
    ):
        raise ValueError("formal parent/sub-issue relations are inconsistent")
    if managed and parent.parent is not None:
        raise ValueError("an existing child must be managed by its formal parent")
    # A nested manual parent still manages its children. A leaf belonging to
    # another parent cannot become a standalone target through this entry point.
    items = (
        children if children or parent.parent is not None else {parent.number: parent}
    )
    orphaned = {
        int(n)
        for n, task in tasks.items()
        if int(n) not in items
        and (
            task.get("delegated")
            or task.get("verification")
            or task.get("creation_authorization")
        )
        and not (int(n) == parent.number and stopped_self_work(task))
        # Ended children may have been formally moved to another parent. Keep
        # their history without preventing independent siblings from working.
        # A formerly split parent with *no* remaining children never falls back
        # to implementing itself from that history.
        and not (children and int(n) != parent.number and worker_ended(task))
    }
    orphaned.update(
        n
        for n in (reviewed_targets or set()) - set(items)
        if n == parent.number or parent.number in items
    )
    if orphaned:
        raise WorkTargetsChanged(
            "work targets changed with existing execution/results: "
            + ", ".join(f"#{n}" for n in sorted(orphaned))
            + "; stop the existing worker and explicitly reconcile the split/relations "
            "and PR ownership before starting other work; saved evidence is retained"
        )
    return items


def require_separate_self_prs(
    parent: Issue, pulls: Sequence[Mapping[str, Any]], repo: str
) -> None:
    """Check already collected PRs at execution/acceptance boundaries.

    Resolving formal targets for a read-only, per-PR diagnostic must not query
    unrelated PR bodies or erase other PRs' facts on a retrieval failure.
    """
    if (
        parent.parent is None
        and parent.children
        and any(
            parent.number in closing_issues(pr.get("body") or "", repo) for pr in pulls
        )
    ):
        raise WorkTargetsChanged(
            "work targets changed with an existing PR closing the parent itself; "
            "explicitly reconcile the split and PR ownership before starting other work"
        )


def dependency_graph(
    parent: Issue, items: dict[int, Issue]
) -> tuple[dict[int, Issue], list[int]]:
    """Separate self-work's external dependencies from a formal child DAG.

    The substituted body is only a policy input; requirements and published
    bodies always use the original Issue. External dependencies are checked by
    the caller against GitHub, never admitted into the implementation targets.
    """
    if parent.number not in items:
        return items, []
    deps = dependencies(parent.body)
    if parent.number in deps:
        raise ValueError("standalone issue depends on itself")
    return {
        parent.number: replace(
            parent, body=replace_section(parent.body, "依存関係", "なし。")
        )
    }, deps
