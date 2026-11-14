"""Read-only publication inventory and one-pass replay of captured worker turns."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

from . import journal, schema
from .journal import PublicationError, require


class RemainingPublication(TypedDict):
    kind: Literal["worker_log", "decision_log"]
    parent: int
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


def worker_paths(repo: str, parent: int | None, *, state_dir: Path) -> list[Path]:
    require(
        bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)),
        "record_repository",
    )
    require(parent is None or schema.integer(parent), "invalid_log_reference")
    root = Path(state_dir) / "worker-logs" / schema.sha(repo.casefold().encode())
    require(
        not any(p.is_symlink() for p in (root, *root.parents)),
        "private_storage_symlink",
    )
    return sorted(
        root.glob(f"{parent if parent is not None else '*'}/*/*/*/journal.json")
    )


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
        and source["repo"].casefold() == repo.casefold()
        and path.parts[-5:-1]
        == (
            str(source["parent"]),
            str(source["issue"]),
            source["public_key"],
            str(source["sequence"]),
        ),
        "journal_identity_changed",
    )
    journal.view(row)
    return dict(row)


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
    Corrupt storage raises PublicationError instead of implying an empty queue.
    """
    from decision_logs.capture import status as capture_status
    from decision_logs.publication_state import read as publication_read

    result: list[RemainingPublication] = []
    for path in worker_paths(repo, parent, state_dir=state_dir):
        row = worker_row(path, repo)
        if row["state"] in {"merged", "legacy"}:
            continue
        source = row["source"]
        result.append(
            {
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
        )
    log = decision_journal(repo, parent, state_dir=state_dir)
    statuses = log.status()
    count = sum(r["publication_status"] != "published" for r in statuses)
    incomplete_capture = (
        capture_status(state_dir, log.repo, parent)["status"] != "captured"
    )
    document = publication_read(
        log.path.with_name("publication.json"), log.repo, parent
    )
    require_batch_captures(log, document)
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


def retryable_rejection(row: dict[str, Any]) -> bool:
    return bool(
        row["state"] == "blocked"
        and row.get("stage") == "merge_rejected"
        and row.get("merge_http_status") == 405
        and row.get("error") == "merge_conflict"
        and row.get("merge_attempted") is False
        and "files" in row
    )


def continue_workers(
    repo: str,
    parent: int | None,
    *,
    state_dir: Path,
    dry_run: bool = False,
    exclude: set[tuple[int, str, int]] | None = None,
    only: set[tuple[int, str, int]] | None = None,
    owner_check: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """One visit per existing journal; never allocates a key or starts a worker."""
    from dispatcher.runtime import Runtime
    from parent.manual_workers import RUNNER

    from . import service

    results = []
    for path in worker_paths(repo, parent, state_dir=state_dir):
        try:
            row = worker_row(path, repo)
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
            retry = retryable_rejection(row)
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
                require(owner_check is None or owner_check(), "log_owner_mismatch")
                if row["state"] == "captured" and src.get("run_directory"):
                    runtime = Runtime(RUNNER)
                    proof = runtime.inspect(Path(src["run_directory"]))
                    if runtime.ended(proof):
                        value = service.finish(path, proof, publish_now=False)
                if value["state"] in {"ready", "pending"} or retry:
                    require(owner_check is None or owner_check(), "log_owner_mismatch")
                    value = service.publish(path, retry=retry)
            results.append(
                {
                    "parent": src["parent"],
                    "issue": src["issue"],
                    **value,
                    "action": action,
                    "dry_run": dry_run,
                }
            )
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            results.append(
                {
                    "state": "blocked",
                    "error": str(exc)
                    if isinstance(exc, PublicationError)
                    else "journal_unavailable",
                    "saved_to_main": False,
                }
            )
            if isinstance(exc, PublicationError) and str(exc) == "log_owner_mismatch":
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

    def owner_is_current() -> bool:
        with store.operation(f"{repo.casefold()}#{parent}"), store.locked():
            current = store.parents.get(f"{repo.casefold()}#{parent}", {})
            return bool(
                current.get("generation") == generation
                and current.get("session") == identity
            )

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
