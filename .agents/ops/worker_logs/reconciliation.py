"""Read-only publication inventory and one-pass replay of captured worker turns."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from . import journal, schema
from .journal import PublicationError, require
from .ownership import publication_owner, require_owner


class RemainingPublication(TypedDict):
    kind: Literal["worker_log", "decision_log"]
    parent: int | None
    issue: int | None
    public_key: str | None
    sequence: int | None
    state: str
    stage: str | None
    error: str | None
    pr_number: int | None
    pr_head: str | None
    unpublished_count: int | None
    batch_key: str | None
    next_action: NotRequired[str]
    journal_reason: NotRequired[str]
    journal_identity: NotRequired[Literal["path_only", "unknown"]]
    namespace_sha256: NotRequired[str]


def decision_resume_command(repo: str, parent: int, *, assigned: bool) -> str:
    """A public resume hint, without changing the remaining-inventory schema."""
    actor = str(parent) if assigned else "RECOVERY_PARENT"
    suffix = "" if assigned else f" --unassigned-parent {parent}"
    return f"parent --repo {repo} --parent {actor} publish-decision-log{suffix}"


def decision_journal(repo: str, parent: int, *, state_dir: Path) -> Any:
    """Repository aliases share a namespace; retain the producer's spelling."""
    from decision_logs.journal import Journal

    log = Journal(Path(state_dir), repo, parent)
    require(
        not any(p.is_symlink() for p in (log.path, *log.path.parents)),
        "private_storage_symlink",
    )
    if log.path.exists():
        header = journal.decode(log.path.read_bytes())
        require(
            isinstance(header.get("repo"), str)
            and header["repo"].casefold() == repo.casefold()
            and header.get("parent") == parent,
            "journal_identity_changed",
        )
        log = Journal(Path(state_dir), header["repo"], parent)
    return log


def worker_root(repo: str, state_dir: Path) -> Path:
    require(
        bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)),
        "record_repository",
    )
    digest: str = schema.sha(repo.casefold().encode())
    return state_dir / "worker-logs" / digest


def _journal_paths(
    path: Path, depth: int, *, namespace: bool = False
) -> Iterator[tuple[Path, str | None]]:
    """Walk the fixed namespace without glob suppressing unreadable directories."""
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            yield path, "private_storage_symlink"
        elif not stat.S_ISDIR(mode):
            if namespace:
                yield path, "journal_read_failed"
        elif depth:
            for child in sorted(path.iterdir()):
                # Parent, issue and turn are positive numbers. The delegation
                # key is the only hex directory. Sidecars such as publication.lock
                # and batches are outside the journal namespace.
                expected = (
                    schema.digest(child.name, 32)
                    if depth == 2
                    else _namespace_number(child.name) is not None
                )
                yield from _journal_paths(child, depth - 1, namespace=expected)
        else:
            path = path / "journal.json"
            path.lstat()
            yield path, None
    except FileNotFoundError:
        return
    except OSError as exc:
        yield path, journal.unavailable_reason(exc)


def _worker_paths(root: Path, parent: int | None) -> Iterator[tuple[Path, str | None]]:
    require(parent is None or schema.integer(parent), "invalid_log_reference")
    try:
        if any(p.is_symlink() for p in (root, *root.parents)):
            yield root, "private_storage_symlink"
            return
    except OSError as exc:
        yield root, journal.unavailable_reason(exc)
        return
    if parent is None:
        yield from _journal_paths(root, 4, namespace=True)
    else:
        yield from _journal_paths(root / str(parent), 3, namespace=True)


def worker_paths(repo: str, parent: int | None, *, state_dir: Path) -> list[Path]:
    paths = []
    for path, reason in _worker_paths(worker_root(repo, state_dir), parent):
        require(reason is None, "journal_unavailable")
        paths.append(path)
    return paths


def worker_row(path: Path, repo: str) -> dict[str, Any]:
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "private_storage_symlink",
    )
    row = journal.read(path)
    source = row["source"]
    require(
        schema.integer(source.get("parent"))
        and schema.integer(source.get("issue"))
        and (row.get("head") is None or schema.digest(row["head"], 40))
        and isinstance(source.get("repo"), str)
        and source["repo"].casefold() == repo.casefold()
        and path.parts[-5:-1]
        == (
            str(source["parent"]),
            str(source["issue"]),
            source.get("public_key"),
            str(source.get("sequence")),
        ),
        "journal_identity_changed",
    )
    journal.view(row)
    return dict(row)


def _worker_entries(
    root: Path, parent: int | None, repo: str
) -> Iterator[tuple[Path, dict[str, Any], str | None]]:
    for path, reason in _worker_paths(root, parent):
        if reason is None:
            try:
                row = worker_row(path, repo)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                reason = journal.unavailable_reason(exc)
            else:
                yield path, row, None
                continue
        yield path, {}, reason


def _namespace_number(value: str) -> int | None:
    return int(value) if re.fullmatch(r"[1-9][0-9]*", value) else None


def _unavailable_journal(
    relative_path: Path,
    reason: str,
    *,
    parent: int | None,
) -> RemainingPublication:
    """Describe unreadable storage using bounded identifiers and fixed reasons."""
    kind: Literal["worker_log", "decision_log"] = (
        "worker_log" if relative_path.parts[0] == "worker-logs" else "decision_log"
    )
    identified = True
    if kind == "worker_log":
        parts = relative_path.parts[2:]
        path_parent = _namespace_number(parts[0]) if parts else None
        parent = path_parent or parent
        identified = (
            len(parts) == 5
            and parts[4] == "journal.json"
            and path_parent is not None
            and _namespace_number(parts[1]) is not None
            and schema.digest(parts[2], 32)
            and _namespace_number(parts[3]) is not None
        )
    return {
        "kind": kind,
        "parent": parent,
        "issue": None,
        "public_key": None,
        "sequence": None,
        "state": "blocked",
        "stage": None,
        "error": "journal_unavailable",
        "pr_number": None,
        "pr_head": None,
        "unpublished_count": None,
        "batch_key": None,
        "journal_reason": reason,
        "journal_identity": "path_only" if identified else "unknown",
        "namespace_sha256": schema.sha(os.fsencode(relative_path)),
    }


def require_batch_captures(log: Any, document: dict[str, Any]) -> None:
    """An unfinished batch must retain its original prepared capture envelopes."""
    batches = [b for b in document["batches"] if b["state"] != "merged"]
    if not batches:
        return
    captured = {p["record"]["id"]: p for p in log.public_payloads()}
    require(
        all(
            captured.get(payload["record"]["id"]) == payload
            for batch in batches
            for payload in batch["payloads"]
        ),
        "input_capture_failed",
    )


def remaining_publications(
    repo: str, parent: int, *, state_dir: Path
) -> list[RemainingPublication]:
    """No locks, capture adapters, subprocesses, network, or state creation.

    public_key is the existing d-hex key for worker logs, the batch key for
    decision logs, or None if no batch exists. Unknown/missing values stay None.
    Unreadable storage stays visible as blocked inventory with an unknown count.
    """
    from decision_logs.capture import status as capture_status
    from decision_logs.journal import Journal
    from decision_logs.publication_state import read as publication_read

    result: list[RemainingPublication] = []
    root = worker_root(repo, state_dir)
    for path, row, reason in _worker_entries(root, parent, repo):
        if reason is not None:
            result.append(
                _unavailable_journal(path.relative_to(state_dir), reason, parent=parent)
            )
            continue
        if row["state"] in {"merged", "legacy"}:
            continue
        source = row["source"]
        item: RemainingPublication = {
            "kind": "worker_log",
            "parent": parent,
            "issue": source["issue"],
            "public_key": "d-" + source["public_key"],
            "sequence": source["sequence"],
            "state": row["state"],
            "stage": row.get("stage"),
            "error": row.get("error"),
            "pr_number": row.get("pr_number"),
            "pr_head": row.get("head"),
            "unpublished_count": None,
            "batch_key": None,
        }
        action = journal.retry_action(row)
        if action is not None:
            item["next_action"] = action
        result.append(item)

    decision_path: Path | None = None
    try:
        log = decision_journal(repo, parent, state_dir=state_dir)
        decision_path = log.path
        statuses = log.status()
        decision_path = log.path.with_name("publication.json")
        document = publication_read(decision_path, log.repo, parent)
        decision_path = log.path
        require_batch_captures(log, document)
        capture = capture_status(state_dir, log.repo, parent)
        require(capture.get("error") != "journal_unavailable", "journal_unavailable")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        if isinstance(exc, PublicationError) and str(exc) == "input_capture_failed":
            raise
        if decision_path is None:
            decision_path = Journal(state_dir, repo, parent).path
        result.append(
            _unavailable_journal(
                decision_path.relative_to(state_dir),
                journal.unavailable_reason(exc),
                parent=parent,
            )
        )
        return result
    count = sum(r["publication_status"] != "published" for r in statuses)
    incomplete_capture = capture["status"] != "captured"
    batch = next((b for b in document["batches"] if b["state"] != "merged"), None)
    if count or batch or incomplete_capture:
        revisions = batch["revisions"] if batch else []
        blocked = next(
            (r["error"] for r in statuses if r["publication_status"] == "blocked"), None
        )
        result.append(
            {
                "kind": "decision_log",
                "parent": parent,
                "issue": None,
                "public_key": batch["key"] if batch else None,
                "sequence": None,
                "state": batch["state"]
                if batch
                else "captured"
                if incomplete_capture
                else "blocked"
                if blocked
                else "unprepared",
                "stage": batch["stage"]
                if batch
                else "capture_incomplete"
                if incomplete_capture
                else "awaiting_publication",
                "error": (batch["error"] if batch else blocked)
                or ("input_capture_failed" if incomplete_capture else None),
                "pr_number": batch["pr_number"] if batch else None,
                "pr_head": revisions[-1]["head"] if revisions else None,
                "unpublished_count": None if incomplete_capture else count,
                "batch_key": batch["key"] if batch else None,
            }
        )
    return result


def continue_workers(
    repo: str,
    parent: int | None,
    *,
    state_dir: Path,
    dry_run: bool = False,
    exclude: set[tuple[int, str, int]] | None = None,
    only: set[tuple[int, str, int]] | None = None,
    milestone: bool = True,
    owner_check: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """One visit per existing journal; never allocates a key or starts a worker."""
    from dispatcher.runtime import Runtime
    from parent.manual_workers import RUNNER

    from . import service

    results = []
    root = worker_root(repo, state_dir)
    for path, row, reason in _worker_entries(root, parent, repo):
        if reason is not None:
            results.append(
                {
                    **_unavailable_journal(
                        path.relative_to(state_dir), reason, parent=parent
                    ),
                    "saved_to_main": False,
                }
            )
            continue
        try:
            src = row["source"]
            key = (src["issue"], "d-" + src["public_key"], src["sequence"])
            if (
                exclude
                and key in exclude
                or only is not None
                and key not in only
                or row["state"] in {"merged", "legacy"}
            ):
                continue
            if row.get("publication_mode") == "delegation_batch" and not milestone:
                continue
            retry = journal.retryable_rejection(row)
            value = journal.view(row)
            action = (
                "inspect_original_run"
                if row["state"] == "captured"
                else (
                    "continue_publication"
                    if row["state"] in {"ready", "pending"} or retry
                    else "retain_failure"
                )
            )
            if not dry_run:
                require_owner(owner_check)
                if row["state"] == "captured" and src.get("run_directory"):
                    runtime = Runtime(RUNNER)
                    proof = runtime.inspect(Path(src["run_directory"]))
                    if runtime.ended(proof):
                        value = service.finish(path, proof, publish_now=False)
                if value["state"] in {"ready", "pending"} or retry:
                    value = service.publish(
                        path,
                        retry=retry,
                        owner_check=owner_check,
                        only_turn=only is not None,
                    )
            if "journal_reason" in value:
                value.update(
                    _unavailable_journal(
                        path.relative_to(state_dir),
                        value["journal_reason"],
                        parent=parent,
                    )
                )
            results.append(
                {
                    "parent": src["parent"],
                    "issue": src["issue"],
                    **value,
                    "action": action,
                    "dry_run": dry_run,
                }
            )
            if value.get("error") == "log_owner_mismatch":
                break
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            code = (
                str(exc) if isinstance(exc, PublicationError) else "journal_unavailable"
            )
            value = {"state": "blocked", "error": code, "saved_to_main": False}
            if code == "journal_unavailable":
                value.update(
                    _unavailable_journal(
                        path.relative_to(state_dir),
                        journal.unavailable_reason(exc),
                        parent=parent,
                    )
                )
            results.append(value)
            if code == "log_owner_mismatch":
                break
    return results


def bulk_command(
    repo: str, parent: int, identity: str | None, *, store: Any, args: Any, root: Path
) -> dict[str, Any]:
    """The assigned recovery parent may replay foreign, immutable journals."""
    sources = args.source_parent or []
    require(all(schema.integer(p) for p in sources), "invalid_log_reference")
    require(not args.pr or args.adopt_orphan_prs, "invalid_log_reference")
    require(args.turn is None and args.delegation_key is None, "invalid_log_reference")
    generation = None
    if not args.dry_run:
        with store.operation(f"{repo.casefold()}#{parent}"), store.locked():
            owner = store.parents.get(f"{repo.casefold()}#{parent}", {})
            require(owner.get("session") == identity, "log_owner_mismatch")
            require(
                owner.get("status") == "active"
                or (
                    not args.adopt_orphan_prs
                    and sources == [parent]
                    and owner.get("status") in {"ended", "released", "handoff_pending"}
                ),
                "log_owner_mismatch",
            )
            generation = owner["generation"]

    owner_is_current = publication_owner(store, repo, parent, identity, generation)

    if args.adopt_orphan_prs:
        from .orphans import recover_orphans

        require(not sources, "invalid_log_reference")
        logs = recover_orphans(
            repo,
            state_dir=store.directory,
            root=root,
            numbers=args.pr,
            dry_run=args.dry_run,
            owner_check=owner_is_current,
        )
    else:
        logs = []
        for selected in sorted(set(sources)) if sources else [None]:
            logs.extend(
                continue_workers(
                    repo,
                    selected,
                    state_dir=store.directory,
                    dry_run=args.dry_run,
                    owner_check=owner_is_current,
                )
            )
            if any(row.get("error") == "log_owner_mismatch" for row in logs):
                break
    value = {"recovery_parent": parent, "logs": logs, "dry_run": args.dry_run}
    if not args.dry_run:
        with store.operation(f"{repo.casefold()}#{parent}"), store.locked():
            current = store.parents.get(f"{repo.casefold()}#{parent}", {})
            value["generation"] = generation
            if (
                current.get("generation") != generation
                or current.get("session") != identity
            ):
                value["error"] = "log_owner_mismatch"
    with store.snapshot():
        owners = dict(store.parents)
    for value_row in logs:
        source_parent = value_row.get("parent")
        value_row["owner_status"] = owners.get(
            f"{repo.casefold()}#{source_parent}", {}
        ).get("status", "unassigned")
    if not args.dry_run:
        target = (
            store.directory
            / "publication-recovery"
            / schema.sha(repo.casefold().encode())
            / f"{parent}.json"
        )
        with journal.locked(target):
            journal.atomic(target, value)
    return value
