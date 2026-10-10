"""Pure classification of local branches / worktrees into cleanup categories.

This module has no side effects: ``inventory.py`` gathers the facts (git, gh,
/proc, training queue) and this module turns one :class:`Facts` record into a
category plus the reasons behind it. The policy is documented in README.md.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Literal

Category = Literal["auto_delete", "approval_required", "protected"]
ReasonKind = Literal["protect", "approval", "auto", "info"]

# Every reason code and the category it pushes an entry towards. The report JSON
# exposes ``code``; README.md documents the meaning of each.
REASON_KINDS: dict[str, ReasonKind] = {
    # protect: never deleted, not even with approval
    "main_worktree": "protect",
    "protected_branch": "protect",
    "current_checkout": "protect",
    "protected_worktree": "protect",
    "worktree_locked": "protect",
    "worktree_missing": "protect",
    "process_using_worktree": "protect",
    "queue_job_references": "protect",
    "open_pr": "protect",
    "undetermined": "protect",
    # approval: deleted only by an explicit `delete <id>` command
    "no_pr": "approval",
    "pr_closed_unmerged": "approval",
    "commits_after_merge": "approval",
    "diverged_from_merged_pr": "approval",
    "detached_head": "approval",
    "uncommitted_changes": "approval",
    "untracked_files": "approval",
    "ignored_files": "approval",
    "submodule_populated": "approval",
    "recently_active": "approval",
    # auto: evidence that everything on the branch is merged
    "merged_pr": "auto",
    "ancestor_of_default_branch": "auto",
    # info: shown for context, does not change the category
    "stale": "info",
}

MergeRelation = Literal[
    "equal", "tip_behind_head", "tip_ahead", "diverged", "head_missing"
]


@dataclass(frozen=True)
class Reason:
    code: str
    detail: str

    def __post_init__(self) -> None:
        if self.code not in REASON_KINDS:
            raise ValueError(f"unknown reason code: {self.code}")

    @property
    def kind(self) -> ReasonKind:
        return REASON_KINDS[self.code]


@dataclass(frozen=True)
class MergedPr:
    number: int
    head_oid: str
    relation: MergeRelation
    commits_after: int | None = None  # set when relation == "tip_ahead"


@dataclass(frozen=True)
class WorktreeState:
    """``git status`` digest of one worktree (paths relative to the worktree)."""

    dirty_tracked: tuple[str, ...]
    untracked: tuple[str, ...]  # non-disposable untracked entries
    ignored: tuple[str, ...]  # non-disposable ignored entries
    disposable_untracked: tuple[str, ...]  # symlinks / config-matched untracked entries
    disposable_ignored: tuple[str, ...]  # symlinks / config-matched ignored entries
    submodule_populated: tuple[str, ...]


@dataclass(frozen=True)
class Facts:
    branch: str | None
    tip: str | None  # branch tip, or worktree HEAD for a detached worktree
    worktree: str | None
    is_main_worktree: bool = False
    worktree_locked: str | None = None  # lock reason ("" when locked without one)
    worktree_missing: bool = False
    worktree_state: WorktreeState | None = None
    protected_branch: bool = False
    protected_worktree: bool = False
    current_checkout: str | None = None  # why it counts as the current checkout
    process_refs: tuple[str, ...] = ()
    queue_refs: tuple[str, ...] = ()
    open_prs: tuple[int, ...] = ()
    closed_prs: tuple[int, ...] = ()
    merged_prs: tuple[MergedPr, ...] = ()
    ancestor_of: str | None = None  # default-branch ref that contains the tip
    idle_hours: float = 0.0
    undetermined: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Policy:
    stale_days: float
    min_idle_hours: float


@dataclass(frozen=True)
class Decision:
    category: Category
    reasons: tuple[Reason, ...]


def _short(items: tuple[str, ...], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _merge_reasons(facts: Facts) -> tuple[list[Reason], bool]:
    """Return merge-related reasons and whether the tip is fully merged."""
    reasons: list[Reason] = []
    by_relation: dict[str, list[MergedPr]] = {}
    for pr in facts.merged_prs:
        by_relation.setdefault(pr.relation, []).append(pr)

    merged = by_relation.get("equal", []) + by_relation.get("tip_behind_head", [])
    if merged:
        pr = merged[0]
        how = (
            "tip == PR head"
            if pr.relation == "equal"
            else "tip is an ancestor of PR head"
        )
        reasons.append(Reason("merged_pr", f"PR #{pr.number} merged ({how})"))
        return reasons, True
    if facts.ancestor_of is not None:
        reasons.append(
            Reason(
                "ancestor_of_default_branch", f"tip is contained in {facts.ancestor_of}"
            )
        )
        return reasons, True
    if "tip_ahead" in by_relation:
        pr = by_relation["tip_ahead"][0]
        reasons.append(
            Reason(
                "commits_after_merge",
                f"{pr.commits_after} commit(s) after merged PR #{pr.number} head",
            )
        )
    elif "diverged" in by_relation:
        pr = by_relation["diverged"][0]
        reasons.append(
            Reason(
                "diverged_from_merged_pr",
                f"tip diverged from merged PR #{pr.number} head",
            )
        )
    elif "head_missing" in by_relation:
        pr = by_relation["head_missing"][0]
        reasons.append(
            Reason(
                "undetermined",
                f"merged PR #{pr.number} head {pr.head_oid[:12]} is not in the local object store "
                "(git fetch to decide)",
            )
        )
    elif facts.open_prs:
        pass  # unmerged work under review; `open_pr` already protects it
    elif facts.closed_prs:
        reasons.append(
            Reason(
                "pr_closed_unmerged",
                "closed without merge: " + ", ".join(f"#{n}" for n in facts.closed_prs),
            )
        )
    elif facts.branch is None:
        reasons.append(
            Reason(
                "detached_head",
                "detached worktree whose HEAD is not in the default branch",
            )
        )
    else:
        reasons.append(
            Reason(
                "no_pr", "no PR for this branch and tip is not in the default branch"
            )
        )
    return reasons, False


def classify(facts: Facts, policy: Policy) -> Decision:
    reasons: list[Reason] = []
    if facts.is_main_worktree:
        reasons.append(Reason("main_worktree", "main checkout is never removed"))
    if facts.protected_branch:
        reasons.append(
            Reason(
                "protected_branch", f"{facts.branch} is in the protected branch list"
            )
        )
    if facts.current_checkout is not None:
        reasons.append(Reason("current_checkout", facts.current_checkout))
    if facts.protected_worktree:
        reasons.append(
            Reason(
                "protected_worktree",
                f"{facts.worktree} matches protected_worktree_globs",
            )
        )
    if facts.worktree_locked is not None:
        reasons.append(
            Reason(
                "worktree_locked", facts.worktree_locked or "locked (no reason given)"
            )
        )
    if facts.worktree_missing:
        reasons.append(
            Reason(
                "worktree_missing",
                "worktree directory is missing; run `git worktree prune` by hand",
            )
        )
    if facts.process_refs:
        reasons.append(Reason("process_using_worktree", _short(facts.process_refs)))
    if facts.queue_refs:
        reasons.append(Reason("queue_job_references", _short(facts.queue_refs)))
    if facts.open_prs:
        reasons.append(
            Reason("open_pr", "open: " + ", ".join(f"#{n}" for n in facts.open_prs))
        )
    for detail in facts.undetermined:
        reasons.append(Reason("undetermined", detail))

    merge_reasons, fully_merged = _merge_reasons(facts)
    reasons.extend(merge_reasons)

    state = facts.worktree_state
    if state is not None:
        if state.dirty_tracked:
            reasons.append(Reason("uncommitted_changes", _short(state.dirty_tracked)))
        if state.untracked:
            reasons.append(Reason("untracked_files", _short(state.untracked)))
        if state.ignored:
            reasons.append(Reason("ignored_files", _short(state.ignored)))
        if state.submodule_populated:
            reasons.append(
                Reason("submodule_populated", _short(state.submodule_populated))
            )

    idle_days = facts.idle_hours / 24.0
    if fully_merged:
        via_pr = any(r.code == "merged_pr" for r in merge_reasons)
        # A PR merge proves intent; plain ancestry (e.g. a branch freshly cut from
        # main) does not, so it additionally needs the stale threshold.
        required_hours = policy.min_idle_hours if via_pr else policy.stale_days * 24.0
        if facts.idle_hours < required_hours:
            reasons.append(
                Reason(
                    "recently_active",
                    f"idle {facts.idle_hours:.1f}h < {required_hours:.1f}h required for auto deletion",
                )
            )
    if idle_days >= policy.stale_days:
        reasons.append(Reason("stale", f"no activity for {idle_days:.1f} days"))

    kinds = {r.kind for r in reasons}
    category: Category
    if "protect" in kinds:
        category = "protected"
    elif "approval" in kinds or not fully_merged:
        category = "approval_required"
    else:
        category = "auto_delete"
    return Decision(category, tuple(reasons))


def matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)


def is_disposable(relpath: str, patterns: tuple[str, ...]) -> bool:
    """Match a status path (dirs end with '/') by full path or by basename."""
    path = relpath.rstrip("/")
    return matches_any(path, patterns) or matches_any(path.rsplit("/", 1)[-1], patterns)
