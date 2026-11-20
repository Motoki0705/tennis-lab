"""Consolidate the four shared memory layers from a pinned main snapshot."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path, PurePosixPath

from . import publish
from .autodream_plan import DEFAULT_RUNNER_TIMEOUT, Plan, Runner, parse_plan, run_agent
from .autodream_state import (
    STATE_FILE,
    Outcome,
    State,
    preserve_state,
    read_state,
    state_lock,
    write_state,
)
from .git import Git, GitTree
from .layout import (
    DEDICATED_PATH,
    INDEX_NAME,
    MEMORY_DIR,
    SHARED_LAYERS,
    SUMMARY_PATH,
    is_entry_filename,
)
from .operations import apply_operations, validation_error
from .schema import Entry, parse_entry
from .tree import EditedTree
from .validate import AuditReport, LimitUsage, Problem, audit, plan_sync, validate

LAYER_DIRS = tuple(f"{MEMORY_DIR}/{layer}" for layer in SHARED_LAYERS)
MEMORY_PATHS = (SUMMARY_PATH, *(f":(top,glob){path}/*.md" for path in LAYER_DIRS))
PROMOTION_PATHS = ("AGENTS.md", ".agents/README.md", ".agents/skills")
NEAR_LIMIT_RATIO = 0.8


@dataclass(frozen=True)
class Thresholds:
    days: int = 7
    main_commits: int = 10
    memory_commits: int = 3

    def __post_init__(self) -> None:
        if min(self.days, self.main_commits, self.memory_commits) < 0:
            raise ValueError("AutoDream thresholds must be nonnegative")


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Trigger:
    reasons: tuple[str, ...]
    elapsed_seconds: float | None
    main_commits: int
    memory_commits: int
    audit_problems: int
    broken_citations: int
    stale_entries: int
    limits_exceeded: int
    generated_drift: int


def decide_trigger(
    *,
    elapsed: timedelta | None,
    main_commits: int,
    memory_commits: int,
    report: AuditReport,
    generated_drift: int,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    force: bool = False,
) -> Trigger:
    broken = len(report.broken_citations)
    stale = len(report.stale)
    exceeded = sum(limit.exceeded for limit in report.limits)
    reasons = [
        name
        for name, condition in (
            ("forced", force),
            ("broken_citations", broken),
            ("stale_entries", stale),
            ("limits_exceeded", exceeded),
            ("generated_drift", generated_drift),
        )
        if condition
    ]
    # The first run uses the full first-parent history and has no age minimum.
    if elapsed is None or elapsed >= timedelta(days=thresholds.days):
        if main_commits >= thresholds.main_commits:
            reasons.append("main_commits")
        if memory_commits >= thresholds.memory_commits:
            reasons.append("memory_commits")
    return Trigger(
        tuple(reasons),
        elapsed.total_seconds() if elapsed is not None else None,
        main_commits,
        memory_commits,
        len(report.problems),
        broken,
        stale,
        exceeded,
        generated_drift,
    )


@dataclass(frozen=True)
class ObservedEntry:
    raw: bytes
    digest: str


@dataclass(frozen=True)
class Snapshot:
    base: str
    tree: GitTree
    entries: dict[str, ObservedEntry]


def orient(git: Git, remote: str) -> Snapshot:
    base = git.fetch_main(remote)
    tree = GitTree(git, base)
    entries = {}
    for directory in LAYER_DIRS:
        if tree.kind(directory, follow_links=False) != "directory":
            continue
        for name in tree.children(directory):
            path = f"{directory}/{name}"
            if tree.kind(path, follow_links=False) == "file" and is_entry_filename(
                name
            ):
                raw = tree.read_bytes(path)
                entries[path] = ObservedEntry(raw, sha256(raw).hexdigest())
    for path in (
        SUMMARY_PATH,
        *(f"{directory}/{INDEX_NAME}" for directory in LAYER_DIRS),
    ):
        if tree.kind(path, follow_links=False) == "file":
            tree.read_bytes(path)
    return Snapshot(base, tree, entries)


@dataclass(frozen=True)
class Signals:
    audit: AuditReport
    generated_drift: tuple[str, ...]
    sync_problems: list[Problem]
    near_limits: tuple[LimitUsage, ...]
    changed_citations: dict[str, list[str]]
    promotion_paths: list[str]
    duplicate_candidates: list[tuple[str, ...]]
    main_commits: int
    memory_commits: int


def _entry_metadata(snapshot: Snapshot, today: date) -> list[Entry]:
    entries = []
    for path, observed in snapshot.entries.items():
        _, layer, filename = path.rsplit("/", 2)
        try:
            entries.append(
                parse_entry(layer, filename, observed.raw.decode("utf-8"), today=today)
            )
        except ValueError:
            # Audit reports invalid entries. They cannot provide structured metadata.
            continue
    return entries


def _citation_paths(tree: GitTree, entries: list[Entry]) -> dict[str, set[str]]:
    citations: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for citation in entry.citations:
            path = PurePosixPath(citation.path).as_posix()
            if path in (".", DEDICATED_PATH) or path.startswith(f"{DEDICATED_PATH}/"):
                continue
            paths = {path}
            # Audit reports resolution errors. The lexical path can still be in history.
            with suppress(ValueError, FileNotFoundError, NotADirectoryError):
                paths.add(tree.resolve(path))
            citations[entry.path].update(
                path
                for path in paths
                if path not in (".", DEDICATED_PATH)
                and not path.startswith(f"{DEDICATED_PATH}/")
            )
    return dict(citations)


def _duplicates(entries: list[Entry]) -> list[tuple[str, ...]]:
    groups = defaultdict(list)
    for entry in entries:
        for field, value in (("title", entry.title), ("summary", entry.summary)):
            groups[field, " ".join(value.casefold().split())].append(entry.path)
    return sorted({tuple(sorted(group)) for group in groups.values() if len(group) > 1})


def gather_signal(
    git: Git, snapshot: Snapshot, previous: str | None, *, today: date
) -> Signals:
    base = snapshot.base
    if previous is not None and not git.is_ancestor(previous, base):
        raise ValueError(
            "the last consolidated commit is no longer an ancestor of remote main"
        )
    history = base if previous is None else f"{previous}..{base}"
    count = ("rev-list", "--first-parent", "--count", history, "--")
    main_commits = int(git.run(*count))
    memory_commits = int(git.run(*count, *MEMORY_PATHS))
    report = audit(snapshot.tree, today=today)
    drift, sync_problems = plan_sync(snapshot.tree, today=today)
    entries = _entry_metadata(snapshot, today)
    citations = _citation_paths(snapshot.tree, entries)
    sources = set(PROMOTION_PATHS).union(*citations.values())
    raw_paths = git.run(
        "log",
        "--first-parent",
        "--diff-merges=first-parent",
        "--format=",
        "--name-only",
        "--no-renames",
        "-z",
        history,
        "--",
        *(f":(top,literal){path}" for path in sorted(sources)),
    )
    changed = {os.fsdecode(path) for path in raw_paths.split(b"\0") if path}
    return Signals(
        report,
        tuple(sorted(drift)),
        sync_problems,
        tuple(limit for limit in report.limits if NEAR_LIMIT_RATIO <= limit.ratio <= 1),
        {
            path: sorted(paths & changed)
            for path, paths in citations.items()
            if paths & changed
        },
        sorted(
            path
            for path in changed
            if path in PROMOTION_PATHS or path.startswith(".agents/skills/")
        ),
        _duplicates(entries),
        main_commits,
        memory_commits,
    )


def consolidate(
    git: Git,
    snapshot: Snapshot,
    signals: Signals,
    today: date,
    runner: Runner,
    log_dir: Path,
    runner_timeout: float,
) -> Plan:
    context: dict[str, object] = {
        "today": today.isoformat(),
        "entry_sha256": {
            path: entry.digest for path, entry in snapshot.entries.items()
        },
        "signals": asdict(signals),
    }
    plan = parse_plan(
        runner(git, snapshot.base, context, log_dir=log_dir, timeout=runner_timeout)
    )
    for operation in plan["operations"]:
        if operation["op"] == "add":
            continue
        path = f"{MEMORY_DIR}/{operation['layer']}/{operation['slug']}.md"
        if path not in snapshot.entries:
            raise ValueError("update/delete must name an entry in the pinned snapshot")
        digest = snapshot.entries[path].digest
        if operation.get("expected_sha256", digest) != digest:
            raise ValueError(f"{path}: expected_sha256 differs from the pinned entry")
        operation["expected_sha256"] = digest
    edits = (
        apply_operations(snapshot.tree, plan["operations"], today=today)
        if plan["operations"]
        else plan_sync(snapshot.tree, today=today)[0]
    )
    if problems := validate(
        EditedTree(snapshot.tree, edits), today=today, require_fresh=True
    ):
        raise validation_error(problems)
    return plan


def prune_and_index(
    repo: Path,
    plan: Plan,
    signals: Signals,
    *,
    remote: str,
    dry_run: bool,
    today: date,
    guard: publish.PublicationGuard,
) -> publish.Publication | None:
    if plan["operations"]:
        return publish.publish_operations(
            repo,
            plan["operations"],
            remote=remote,
            dry_run=dry_run,
            today=today,
            require_fresh=True,
            guard=guard,
        )
    if signals.generated_drift:
        reason = (
            plan["summary"]
            if plan["repair_index"]
            else "AutoDream が生成文書のずれを検出したため。"
        )
        return publish.publish_index_repair(
            repo,
            reason=reason,
            remote=remote,
            dry_run=dry_run,
            today=today,
            require_fresh=True,
            guard=guard,
        )
    return None


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _confirmed_merge(
    state: State,
    number: int,
    head: str,
    clock: Callable[[], datetime],
    cleanup_error: str | None,
) -> State:
    return replace(
        state,
        last_outcome="merged",
        last_completed_at=clock().astimezone(UTC).isoformat(),
        last_consolidated_sha=head,
        last_pr=number,
        last_merge_sha=head,
        pending_publication=None,
        branch_cleanup_error=cleanup_error,
    )


def run(
    repo: Path,
    *,
    state_dir: Path,
    remote: str = "origin",
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    force: bool = False,
    dry_run: bool = False,
    runner: Runner = run_agent,
    runner_timeout: float = DEFAULT_RUNNER_TIMEOUT,
    clock: Callable[[], datetime] = _local_now,
) -> dict[str, object]:
    state_dir = state_dir.resolve()
    with state_lock(state_dir):
        state_file = state_dir / STATE_FILE
        started = clock()
        started_at = started.astimezone(UTC).isoformat()
        try:
            state = read_state(state_file)
        except (OSError, ValueError) as exc:
            backup = preserve_state(state_file)
            error = f"Read State: {exc}; original preserved at {backup}; manual recovery required"
            write_state(
                state_file,
                State(
                    last_run_at=started_at,
                    last_outcome="failed",
                    last_error=error,
                    state_backup=str(backup),
                ),
            )
            raise ValueError(error) from exc
        today = started.date()
        state = replace(
            state,
            last_run_at=started_at,
            last_trigger=None,
            last_error=None,
        )
        phase = "Read State"
        report: dict[str, object] = {"dry_run": dry_run}
        outcome: Outcome = "skipped_trigger"
        completion: State | None = None

        def record_intent(intent: publish.PublicationIntent | None) -> None:
            nonlocal state
            state = replace(state, pending_publication=intent)
            if intent is not None:
                state = replace(state, last_outcome="publishing")
                write_state(state_file, state)

        try:
            if state.state_backup is not None:
                raise ValueError(
                    f"manual recovery required; original preserved at {state.state_backup}"
                )
            if not math.isfinite(runner_timeout) or runner_timeout <= 0:
                raise ValueError("runner timeout must be finite and positive")
            git = Git(repo)
            if (pending := state.pending_publication) is not None:
                phase = "Recover Publication"
                cleanup_error = publish.reconcile_publication(
                    git, remote, pending, dry_run=dry_run
                )
                report.update(
                    recovered=not dry_run, confirmed_publication=asdict(pending)
                )
                if dry_run:
                    outcome = "dry_run"
                else:
                    assert pending.pr is not None  # PR identity was confirmed above.
                    completion = _confirmed_merge(
                        state, pending.pr, pending.head, clock, cleanup_error
                    )
            else:
                phase = "Orient"
                snapshot = orient(git, remote)
                report["base"] = snapshot.base
                phase = "Gather Signal"
                signals = gather_signal(
                    git, snapshot, state.last_consolidated_sha, today=today
                )
                previous = state.last_completed_at
                trigger = decide_trigger(
                    elapsed=started - datetime.fromisoformat(previous)
                    if previous
                    else None,
                    main_commits=signals.main_commits,
                    memory_commits=signals.memory_commits,
                    report=signals.audit,
                    generated_drift=len(signals.generated_drift),
                    thresholds=thresholds,
                    force=force,
                )
                state = replace(state, last_trigger=asdict(trigger))
                if dry_run:
                    report.update(signals=asdict(signals), plan=None, validation=None)
                if trigger.reasons:
                    phase = "Consolidate"
                    plan = consolidate(
                        git,
                        snapshot,
                        signals,
                        today,
                        runner,
                        state_dir / "agent-runs",
                        runner_timeout,
                    )
                    phase = "Prune & Index"
                    publication = prune_and_index(
                        repo,
                        plan,
                        signals,
                        remote=remote,
                        dry_run=dry_run,
                        today=today,
                        guard=publish.PublicationGuard(snapshot.base, record_intent),
                    )
                    report["publication"] = asdict(publication) if publication else None
                    if dry_run:
                        outcome = "dry_run"
                        report.update(plan=plan, validation="passed")
                    elif publication is None:
                        completion = replace(
                            state,
                            last_outcome="noop",
                            last_completed_at=clock().astimezone(UTC).isoformat(),
                            last_consolidated_sha=snapshot.base,
                        )
                    else:
                        if publication.pr is None or publication.merge_sha is None:
                            raise ValueError("publisher did not confirm a merge")
                        completion = _confirmed_merge(
                            state,
                            publication.pr,
                            publication.merge_sha,
                            clock,
                            publication.branch_cleanup_error,
                        )
        except Exception as exc:
            error = f"{phase}: {exc}"
            write_state(
                state_file, replace(state, last_outcome="failed", last_error=error)
            )
            raise ValueError(error) from exc
        # Keep the prior completion and intent until the new outcome is durable.
        final_state = completion or replace(state, last_outcome=outcome)
        write_state(state_file, final_state)
        return {**report, **asdict(final_state)}
