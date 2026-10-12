"""Pure classification of local branches and linked worktrees into cleanup categories.

This module has no side effects: ``inventory.py`` gathers the facts (git, gh,
/proc, training queue) and this module turns one :class:`BranchFacts` or
:class:`WorktreeFacts` record into a category plus the reasons behind it. The
policy is documented in README.md.

Scope (see README.md for why): only a *branch* without a worktree can be
``auto_delete``. A worktree is never deleted automatically; at best it is
``approval_required`` and then removable by ``cleanup.py delete worktree:<path>``
when no ``block`` reason applies.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Literal

Category = Literal["auto_delete", "approval_required", "protected"]
ReasonKind = Literal["protect", "block", "approval", "auto", "info"]

# Every reason code and the category it pushes an entry towards. The report JSON
# exposes ``code``; README.md documents the meaning of each.
REASON_KINDS: dict[str, ReasonKind] = {
    # protect: never deleted, not even with approval
    "protected_branch": "protect",
    "current_checkout": "protect",
    "symbolic_ref": "protect",
    "branch_in_use": "protect",
    "open_pr": "protect",
    "protected_worktree": "protect",
    "worktree_locked": "protect",
    "worktree_missing": "protect",
    "operation_in_progress": "protect",
    "process_using_worktree": "protect",
    "queue_job_references": "protect",
    "queue_job_unparseable": "protect",
    "undetermined": "protect",
    # block: the tool refuses until a human resolves it (it never discards work)
    "checked_out_in_worktree": "block",
    "uncommitted_changes": "block",
    "untracked_files": "block",
    "ignored_files": "block",
    "submodule_populated": "block",
    "unpushed_commits": "block",
    # approval: deleted only by an explicit `delete <id>` command
    "worktree_needs_approval": "approval",
    "no_pr": "approval",
    "pr_closed_unmerged": "approval",
    "behind_merged_pr": "approval",
    "commits_after_merge": "approval",
    "diverged_from_merged_pr": "approval",
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
    """``git status --ignored`` digest of one worktree (paths relative to it)."""

    dirty_tracked: tuple[str, ...]
    untracked: tuple[str, ...]
    ignored: tuple[str, ...]
    disposable: tuple[str, ...]  # the top-level `.venv` symlink, when present
    submodule_populated: tuple[str, ...]


@dataclass(frozen=True)
class BranchFacts:
    branch: str
    tip: str
    protected_branch: bool = False
    current_checkout: str | None = None  # checked out in the main checkout
    checked_out_in: tuple[str, ...] = ()  # linked worktrees that check it out
    symbolic_ref: tuple[str, ...] = ()  # it is a symbolic ref or the target of one
    branch_in_use: tuple[str, ...] = ()  # held by a rebase/bisect somewhere
    open_prs: tuple[int, ...] = ()
    closed_prs: tuple[int, ...] = ()
    merged_prs: tuple[MergedPr, ...] = ()
    ancestor_of: str | None = None  # default-branch ref that contains the tip
    idle_hours: float = 0.0
    undetermined: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorktreeFacts:
    worktree: str
    head: str
    branch: str | None  # checked-out branch (None when detached)
    locked: str | None = None  # lock reason ("" when locked without one)
    missing: bool = False
    state: WorktreeState | None = None
    protected_worktree: bool = False
    current_checkout: str | None = None  # contains the invoking working directory
    operations_in_progress: tuple[str, ...] = ()
    process_refs: tuple[str, ...] = ()
    queue_refs: tuple[str, ...] = ()
    queue_unparseable: tuple[str, ...] = ()  # jobs blocking every worktree removal
    pushed: str | None = None  # evidence that HEAD exists outside this machine
    idle_hours: float = 0.0
    undetermined: tuple[str, ...] = ()


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


def _merge_reasons(f: BranchFacts) -> tuple[list[Reason], bool]:
    """Return merge-related reasons and whether the tip is proven merged."""
    by_relation: dict[str, list[MergedPr]] = {}
    for pr in f.merged_prs:
        by_relation.setdefault(pr.relation, []).append(pr)
    if "equal" in by_relation:
        pr = by_relation["equal"][0]
        return [Reason("merged_pr", f"PR #{pr.number} merged; tip == PR head")], True
    if f.ancestor_of is not None:
        return [
            Reason("ancestor_of_default_branch", f"tip is contained in {f.ancestor_of}")
        ], True
    if "tip_behind_head" in by_relation:
        pr = by_relation["tip_behind_head"][0]
        detail = f"tip is an ancestor of merged PR #{pr.number} head"
        return [Reason("behind_merged_pr", detail)], False
    if "tip_ahead" in by_relation:
        pr = by_relation["tip_ahead"][0]
        detail = f"{pr.commits_after} commit(s) after merged PR #{pr.number} head"
        return [Reason("commits_after_merge", detail)], False
    if "diverged" in by_relation:
        pr = by_relation["diverged"][0]
        detail = f"tip diverged from merged PR #{pr.number} head"
        return [Reason("diverged_from_merged_pr", detail)], False
    if "head_missing" in by_relation:
        pr = by_relation["head_missing"][0]
        detail = (
            f"merged PR #{pr.number} head {pr.head_oid[:12]} is not in the local "
            "object store (git fetch to decide)"
        )
        return [Reason("undetermined", detail)], False
    if f.open_prs:
        return [], False  # unmerged work under review; `open_pr` protects it
    if f.closed_prs:
        closed = ", ".join(f"#{n}" for n in f.closed_prs)
        return [Reason("pr_closed_unmerged", f"closed without merge: {closed}")], False
    detail = "no PR for this branch and tip is not in the default branch"
    return [Reason("no_pr", detail)], False


def _category(reasons: list[Reason], auto_allowed: bool) -> Category:
    kinds = {r.kind for r in reasons}
    if "protect" in kinds:
        return "protected"
    if auto_allowed and not kinds & {"block", "approval"}:
        return "auto_delete"
    return "approval_required"


def _stale(idle_hours: float, policy: Policy) -> list[Reason]:
    days = idle_hours / 24.0
    if days >= policy.stale_days:
        return [Reason("stale", f"no activity for {days:.1f} days")]
    return []


def classify_branch(f: BranchFacts, policy: Policy) -> Decision:
    reasons: list[Reason] = []
    if f.protected_branch:
        reasons.append(
            Reason("protected_branch", f"{f.branch} is in the protected branch list")
        )
    if f.current_checkout is not None:
        reasons.append(Reason("current_checkout", f.current_checkout))
    if f.symbolic_ref:
        reasons.append(Reason("symbolic_ref", _short(f.symbolic_ref)))
    if f.branch_in_use:
        reasons.append(Reason("branch_in_use", _short(f.branch_in_use)))
    if f.open_prs:
        opened = ", ".join(f"#{n}" for n in f.open_prs)
        reasons.append(Reason("open_pr", f"open: {opened}"))
    for detail in f.undetermined:
        reasons.append(Reason("undetermined", detail))
    if f.checked_out_in:
        reasons.append(
            Reason(
                "checked_out_in_worktree",
                "remove the worktree first: " + _short(f.checked_out_in),
            )
        )
    merge_reasons, merged = _merge_reasons(f)
    reasons.extend(merge_reasons)
    if merged:
        via_pr = any(r.code == "merged_pr" for r in merge_reasons)
        # A PR merge proves intent; plain ancestry (e.g. a branch freshly cut from
        # main) does not, so it additionally needs the stale threshold.
        required = policy.min_idle_hours if via_pr else policy.stale_days * 24.0
        if f.idle_hours < required:
            reasons.append(
                Reason(
                    "recently_active",
                    f"idle {f.idle_hours:.1f}h < {required:.1f}h required for auto deletion",
                )
            )
    reasons.extend(_stale(f.idle_hours, policy))
    return Decision(_category(reasons, auto_allowed=merged), tuple(reasons))


def classify_worktree(f: WorktreeFacts, policy: Policy) -> Decision:
    reasons: list[Reason] = []
    if f.current_checkout is not None:
        reasons.append(Reason("current_checkout", f.current_checkout))
    if f.protected_worktree:
        reasons.append(
            Reason(
                "protected_worktree", f"{f.worktree} matches protected_worktree_globs"
            )
        )
    if f.locked is not None:
        reasons.append(
            Reason("worktree_locked", f.locked or "locked (no reason given)")
        )
    if f.missing:
        reasons.append(
            Reason(
                "worktree_missing",
                "worktree directory is missing; run `git worktree prune` by hand",
            )
        )
    if f.operations_in_progress:
        reasons.append(
            Reason(
                "operation_in_progress",
                "in progress: " + ", ".join(f.operations_in_progress),
            )
        )
    if f.process_refs:
        reasons.append(Reason("process_using_worktree", _short(f.process_refs)))
    if f.queue_refs:
        reasons.append(Reason("queue_job_references", _short(f.queue_refs)))
    if f.queue_unparseable:
        reasons.append(Reason("queue_job_unparseable", _short(f.queue_unparseable)))
    for detail in f.undetermined:
        reasons.append(Reason("undetermined", detail))

    state = f.state
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
    if f.pushed is None:
        reasons.append(
            Reason(
                "unpushed_commits",
                f"HEAD {f.head[:12]} is in no remote-tracking ref and no PR head",
            )
        )
    reasons.append(
        Reason(
            "worktree_needs_approval",
            "worktrees are removed only by `delete worktree:<path>`"
            + (f" (pushed: {f.pushed})" if f.pushed else ""),
        )
    )
    reasons.extend(_stale(f.idle_hours, policy))
    return Decision(_category(reasons, auto_allowed=False), tuple(reasons))


def matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)
