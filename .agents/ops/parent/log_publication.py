"""One publication continuation boundary shared by parent CLI mutations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from worker_logs.journal import PublicationError, require
from worker_logs.lifecycle import (
    capture_exit,
    publish_decisions,
    record_exit,
    unavailable_exit,
)
from worker_logs.ownership import publication_owner
from worker_logs.reconciliation import (
    continue_workers,
    decision_journal,
    require_batch_captures,
)
from worker_logs.schema import integer

from .state import Store

READ_ONLY = {
    "status",
    "worker-log-status",
    "inspect-worker",
    "decision-log-status",
    "review-inspect",
    "basis-input",
    "basis-check",
    "wait-supplement",
    "wait-worker",
    "wait-merge",
    "merge-status",
    "state-migrate",
}
BEFORE_WORKER = {"run-worker", "start-worker", "continue-worker"}
EXPLICIT_PUBLICATION = {"publish-worker-log", "publish-decision-log"}
MUTATIONS = {
    "register",
    "summary",
    "checking",
    "agree",
    "agreement-ignore-comments",
    "basis-record",
    "basis-revoke",
    "note",
    "notify",
    "decide",
    "create-child",
    "receive-proposal",
    "delegate",
    "configure-delegation",
    "record-worker-run",
    "record-worker-ended",
    "record-result",
    "record-cancellation",
    "decide-worker-plan",
    "refresh-worker-plan",
    "workflow-report",
    "sync",
    "reconcile",
    "stop",
    "resume",
    "stop-worker",
    "improvement",
    "improvement-propose",
    "improvement-extract",
    "improvement-decide",
    "review-request",
    "merge-continue",
    "retry-decision-capture",
    "release",
    "handoff",
    "publish-worker-log",
    "publish-decision-log",
} | BEFORE_WORKER


def read_only(args: Any) -> bool:
    return bool(
        args.command in READ_ONLY
        or getattr(args, "dry_run", False)
        or args.command in {"project-setup", "project-sync"}
        and not args.apply
        or args.command == "evidence"
        and not args.create
    )


def publish_unassigned_decisions(
    args: Any, continuation: Continuation, *, root: Path
) -> dict[str, Any]:
    """An active parent publishes a foreign journal without adopting its work."""
    from decision_logs.publication import Publisher
    from decision_logs.publication_state import read

    store, repo, parent, identity = (
        continuation.store,
        continuation.repo,
        continuation.parent,
        continuation.identity,
    )
    target = args.unassigned_parent
    require(integer(target) and target != parent, "invalid_log_reference")
    target_key = f"{repo.casefold()}#{target}"
    with store.snapshot():
        owner = dict(store.parents.get(continuation.key, {}))
        original_target = dict(store.parents.get(target_key, {}))
    if not (
        identity is not None
        and owner.get("session") == identity
        and owner.get("status") == "active"
        and owner.get("generation") == continuation.generation
        and integer(continuation.generation)
    ):
        raise ValueError(
            "unassigned decision publication requires an active assigned parent"
        )
    if original_target and original_target["status"] not in {"ended", "released"}:
        raise ValueError(
            "decision publication target must be unassigned, not active or handoff_pending"
        )

    def current() -> bool:
        # No ownership is taken and no registry bytes are written. Snapshot is
        # also used on dry-run so inspection creates no operation/registry locks.
        try:
            with store.snapshot():
                actor = store.parents.get(continuation.key, {})
                source = store.parents.get(target_key, {})
                return bool(
                    actor.get("session") == identity
                    and actor.get("status") == "active"
                    and actor.get("generation") == continuation.generation
                    and {k: source.get(k) for k in ("session", "status", "generation")}
                    == {
                        k: original_target.get(k)
                        for k in ("session", "status", "generation")
                    }
                    and (not source or source["status"] in {"ended", "released"})
                )
        except (ValueError, OSError, KeyError, TypeError):
            raise PublicationError("log_owner_mismatch") from None

    log = decision_journal(repo, target, state_dir=store.directory)
    actor = {
        "parent": parent,
        "session": identity,
        "generation": continuation.generation,
    }
    existing = read(log.path.with_name("publication.json"), log.repo, target)
    require_batch_captures(log, existing)
    if args.dry_run:
        value = Publisher(root, log).publish(
            dry_run=True, retry=args.retry, owner_check=current
        )
    else:
        # A fixed old batch is reconciled first. The attribution request may
        # therefore need one following batch. Always allow two visits: another
        # publisher may have created the old batch after the admission read.
        limit = {
            r["id"] for r in log.status() if r["publication_status"] != "published"
        }
        value = {}
        for _ in range(2):
            value = Publisher(root, log, record_ids=limit).publish(
                retry=args.retry, owner_check=current, actor=actor
            )
            if value["state"] != "merged" or value.get("error"):
                break
            pending = {
                r["id"] for r in log.status() if r["publication_status"] != "published"
            }
            if not limit.intersection(pending):
                break
    return {**value, "recovery_parent": parent, "parent": target}


class Continuation:
    def __init__(
        self, args: Any, store: Store, repo: str, parent: int, identity: str | None
    ) -> None:
        self.args, self.store, self.repo, self.parent, self.identity = (
            args,
            store,
            repo,
            parent,
            identity,
        )
        self.done = False
        self.error: str | None = None
        self.generation: int | None = None
        self.key = f"{repo.casefold()}#{parent}"
        self.visited: set[tuple[int, str, int]] = set()
        try:
            self.initial_owner = self._owner_snapshot()
        except (ValueError, OSError, KeyError, TypeError, AttributeError):
            self.initial_owner = {}
        if (
            args.command in EXPLICIT_PUBLICATION
            and self.initial_owner.get("session") == identity
        ):
            # Explicit publication and its postlude share one admission. A
            # same-session registration must not grant that old operation a new epoch.
            self.generation = self.initial_owner.get("generation")

    def _owner_snapshot(self) -> dict[str, Any]:
        with self.store.snapshot():
            return dict(self.store.parents.get(self.key, {}))

    def _publication_owner(self) -> dict[str, Any]:
        with self.store.operation(self.key), self.store.locked():
            return dict(self.store.parents.get(self.key, {}))

    def _same_owner(self, owner: dict[str, Any]) -> bool:
        return (
            self.identity is not None
            and self.generation is not None
            and owner.get("session") == self.identity
            and owner.get("generation") == self.generation
        )

    def recovery_visits(self, value: Any) -> None:
        """Exclude journals already visited by an explicit bulk operation."""
        if not isinstance(value, dict):
            return
        if value.get("parent") == self.parent and all(
            k in value for k in ("issue", "delegation_key", "sequence")
        ):
            self.visited.add(
                (value["issue"], value["delegation_key"], value["sequence"])
            )
        for row in value.get("logs", []):
            self.recovery_visits(row)

    def sweep(self) -> None:
        if self.done or read_only(self.args):
            return
        self.done = True
        if (
            self.args.command == "publish-decision-log"
            and getattr(self.args, "unassigned_parent", None) is not None
        ):
            return
        command = self.args.command
        if command not in MUTATIONS | {"project-setup", "project-sync", "evidence"}:
            return
        if command in EXPLICIT_PUBLICATION and self.generation is None:
            self.error = "log_owner_unavailable"
            return
        try:
            # No lock is held during Git/GitHub publication. Both this short
            # claim and the durable receipt below recheck the current generation.
            with self.store.operation(self.key), self.store.locked():
                owner = self.store.parents.get(self.key, {})
                if self.generation is not None and not self._same_owner(owner):
                    self.error = "log_owner_mismatch"
                    return
                if owner.get("session") != self.identity or owner.get("status") not in (
                    {"active", "ended", "released", "handoff_pending"}
                    if command
                    in {
                        "release",
                        "handoff",
                        "publish-worker-log",
                        "publish-decision-log",
                    }
                    else {"active"}
                ):
                    return
                self.generation = owner["generation"]
            exclude = set(self.visited)
            if command == "publish-worker-log":
                from worker_logs.reconciliation import worker_paths, worker_row

                for path in worker_paths(
                    self.repo, self.parent, state_dir=self.store.directory
                ):
                    row = worker_row(path, self.repo)
                    s = row["source"]
                    if (
                        s["issue"] == getattr(self.args, "child", None)
                        and (self.args.turn is None or s["sequence"] == self.args.turn)
                        and (
                            not self.args.delegation_key
                            or "d-" + s["public_key"] == self.args.delegation_key
                        )
                    ):
                        exclude.add((s["issue"], "d-" + s["public_key"], s["sequence"]))
            values = continue_workers(
                self.repo,
                self.parent,
                state_dir=self.store.directory,
                exclude=exclude,
                owner_check=publication_owner(
                    self.store, self.repo, self.parent, self.identity, self.generation
                ),
            )
            with self.store.operation(self.key), self.store.locked():
                owner = self.store.parents.get(self.key, {})
                if (
                    owner.get("generation") != self.generation
                    or owner.get("session") != self.identity
                ):
                    self.error = "log_owner_mismatch"
                else:
                    receipt = {"logs": values}
                    if (
                        any(v.get("action") != "retain_failure" for v in values)
                        and owner.get("publication_continuation") != receipt
                    ):
                        owner["publication_continuation"] = receipt
                        self.store.save()
            if any(v.get("error") for v in values):
                self.error = self.error or "publication_failed"
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            self.error = (
                str(exc) if isinstance(exc, PublicationError) else "publication_failed"
            )
        if self.error:
            print(
                "parent-session: log publication remains incomplete; inspect worker-log-status",
                file=sys.stderr,
            )

    def complete(self, args: Any, value: Any, *, exit_code: int = 0) -> Any:
        from decision_logs import parent_commands as capture

        self.sweep()
        value = capture.complete(args, value, exit_code=exit_code)
        return self.finish_exit(args, value)

    def failed(self) -> Any:
        """A durable assignment change still needs reconciliation on API failure."""
        if self.args.command not in {"release", "handoff"}:
            self.sweep()
            return None
        observed = None
        try:
            owner = self._owner_snapshot()
            previous = self.initial_owner
            accepting = self.args.command == "handoff" and self.args.accept
            expected = (
                "active"
                if accepting
                else "handoff_pending"
                if self.args.command == "handoff"
                else "ended"
                if self.args.ended
                else "released"
            )
            applied = (
                bool(previous)
                and owner.get("session") == self.identity
                and owner.get("status") == expected
                and (
                    accepting
                    and owner.get("generation") == previous.get("generation", 0) + 1
                    or not accepting
                    and previous.get("session") == self.identity
                    and previous.get("status") == "active"
                    and owner.get("generation") == previous.get("generation")
                )
            )
            if applied:
                # Freeze the saved assignment before publication can overlap a
                # successor's handoff. That handoff must not erase this exit.
                self.generation = owner["generation"]
                observed = {"assignment_observed": expected}
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            AttributeError,
            subprocess.SubprocessError,
        ):
            pass
        self.sweep()
        return self.finish_exit(self.args, observed) if observed is not None else None

    def finish_exit(self, args: Any, value: Any) -> Any:
        if args.command not in {"release", "handoff"} or not isinstance(value, dict):
            return value
        owner: dict[str, Any] = {}
        event = (
            "handoff_accepted"
            if args.command == "handoff" and args.accept
            else "handoff_pending"
            if args.command == "handoff"
            else "ended"
        )
        try:
            owner = self._publication_owner()
            if not self._same_owner(owner):
                self.error = "log_owner_mismatch"
            if self.generation is not None and not capture_exit(
                self.repo,
                self.parent,
                self.generation,
                event,
                state_dir=self.store.directory,
                root=Path.cwd(),
                publication_actor=self.parent
                if owner.get("status") == "active"
                else "unassigned",
                publication_error=self.error,
            ):
                self.error = self.error or "input_capture_failed"
        except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
            self.error = self.error or "input_capture_failed"
        try:
            owner = self._publication_owner()
            if not self._same_owner(owner):
                self.error = "log_owner_mismatch"
            else:
                published = publish_decisions(
                    self.repo,
                    self.parent,
                    state_dir=self.store.directory,
                    root=Path.cwd(),
                    owner_check=lambda: self._same_owner(self._publication_owner()),
                )
                self.error = self.error or published.get("error")
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            self.error = (
                str(exc) if isinstance(exc, PublicationError) else "publication_failed"
            )
        try:
            owner = self._publication_owner()
            if not self._same_owner(owner):
                self.error = "log_owner_mismatch"
            if self.generation is None:
                raise PublicationError("log_owner_mismatch")
            checkpoint = record_exit(
                self.repo,
                self.parent,
                self.generation,
                event,
                state_dir=self.store.directory,
                root=Path.cwd(),
                publication_actor=self.parent
                if owner.get("status") == "active"
                else "unassigned",
                publication_error=self.error,
            )
            return {**value, "publication_exit": checkpoint}
        except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
            return {
                **value,
                "publication_exit": unavailable_exit(
                    self.repo, self.parent, self.generation, event
                ),
            }
