"""Creation-based expiry with durable intent and resumable Git/filesystem removal."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import inventory
from .lifecycle import Stopper
from .storage import (
    LIFETIME_NS,
    CleanupError,
    State,
    locked,
    state_root,
    sync_directory,
    utc,
    write_json,
)


class Cleaner:
    def __init__(
        self,
        cwd: Path,
        *,
        directory: Path | None = None,
        clock: Callable[[], int] = time.time_ns,
        stopper: Stopper | None = None,
    ) -> None:
        self.repo, self.common = inventory.main_checkout(cwd)
        # Repository-scoped state avoids confusing independent clones on a host.
        import hashlib

        key = hashlib.sha256(os.fsencode(self.common)).hexdigest()[:24]
        self.directory = directory or state_root() / "worktree-cleanup" / key
        self.clock = clock
        self.stopper = stopper or Stopper(self.repo)

    def observe(self, state: State, now: int) -> list[dict[str, Any]]:
        rows = inventory.registrations(self.repo, self.common)
        live_ids = set()
        entries = []
        for row in rows:
            previous = next(
                (
                    r
                    for r in state.records.values()
                    if r["phase"] not in {"deleted", "absent"}
                    and not r.get("registration_lost_at")
                    and r["admin"] == row["admin"]
                    and inventory.same_registration(
                        r["admin_identity"], row["admin_identity"]
                    )
                ),
                None,
            )
            if previous is not None:
                row.update(id=previous["id"], admin_identity=previous["admin_identity"])
            key = row["id"]
            if state.records.get(key, {}).get("phase") in {
                "deleted",
                "absent",
            } or state.records.get(key, {}).get("registration_lost_at"):
                # An inode can be recycled after a completed removal on a
                # filesystem without btime; a live registration is a new one.
                row["id"] = key = uuid.uuid4().hex + uuid.uuid4().hex
            live_ids.add(key)
            if key not in state.records:
                born = row["admin_identity"]["birth_time_ns"]
                created = now if born is None else born
                state.records[key] = {
                    **row,
                    "created_at_ns": created,
                    "created_at": utc(created),
                    "source": "first_observed" if born is None else "birth_time",
                    "first_observed_at": utc(now),
                    "phase": "observed",
                }
            record = state.records[key]
            if (
                record["path"] != row["path"] and record["phase"] != "observed"
            ) or record["admin_identity"] != row["admin_identity"]:
                raise CleanupError(
                    "worktree identity/path changed without a new registration"
                )
            if record["phase"] == "observed":
                record.update(row)
            entries.append(self.entry(record, now, row["scope_error"]))
        # Resume interrupted deletion even after Git's registration is gone.
        for key, row in state.records.items():
            if key not in live_ids and row["phase"] == "observed":
                row.update(phase="absent", disappeared_at=utc(now))
            if key not in live_ids and row["phase"] in {
                "stopping",
                "quarantined",
                "unregistered",
            }:
                row.setdefault("registration_lost_at", utc(now))
                entries.append(
                    self.entry(
                        row, now, inventory.scope_error(self.repo, Path(row["path"]))
                    )
                )
        state.save()
        return entries

    @staticmethod
    def entry(row: dict[str, Any], now: int, error: str | None) -> dict[str, Any]:
        deadline = row["created_at_ns"] + LIFETIME_NS
        target = error is None
        return {
            "id": row["id"],
            "path": row["path"],
            "head": row.get("head"),
            "created_at": row["created_at"],
            "source": row["source"],
            "deadline": utc(deadline),
            "decided_at": utc(now),
            "age_hours": (now - row["created_at_ns"]) / (3600 * 10**9),
            "target": target,
            "expired": now >= deadline,
            "selected": target and now >= deadline,
            "scope_error": error,
            "phase": row["phase"],
            **inventory.changes(Path(row.get("quarantine") or row["path"]), error),
        }

    def current(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if row.get("registration_lost_at"):
            return None
        return next(
            (
                r
                for r in inventory.registrations(self.repo, self.common)
                if r["admin"] == row["admin"]
                and inventory.same_registration(
                    r["admin_identity"], row["admin_identity"]
                )
            ),
            None,
        )

    def remove(self, state: State, row: dict[str, Any], now: int) -> None:
        path = Path(row["path"])
        error = inventory.scope_error(self.repo, path)
        if error:
            raise CleanupError(error)
        if row["created_at_ns"] + LIFETIME_NS > now:
            raise CleanupError("worktree has not reached 336 hours")
        quarantine = self.repo / ".claude/worktrees" / (".expired-" + row["id"])
        if row.get("quarantine") not in {None, str(quarantine)}:
            raise CleanupError("invalid persisted quarantine path")
        if quarantine.is_symlink():
            raise CleanupError("quarantine path is a symlink")
        if row["phase"] == "observed":
            live = self.current(row)
            if live is None:
                raise CleanupError(
                    "worktree registration disappeared before expiry intent"
                )
            row.update(
                live, quarantine=str(quarantine), phase="stopping", decided_at=utc(now)
            )
            state.save()
            state.log(row, decided_at=utc(now), phase="intent", deleted=False)
        # Stop attempts are replayed after interruption; stopped is never success.
        live = self.current(row)
        replacement = live is None and (path.exists() or path.is_symlink())
        if not replacement:
            self.stopper.stop(path, utc(now))
        if live is not None and live["path"] != str(path):
            raise CleanupError("worktree registration moved during expiry")
        if row["phase"] == "stopping":
            if quarantine.exists():
                if path.exists():
                    raise CleanupError("original path reappeared beside quarantine")
                inventory.require_directory(quarantine, row["directory_identity"])
            elif path.exists():
                if live is None:
                    raise CleanupError(
                        "registration vanished while original directory remains"
                    )
                inventory.require_directory(path, row["directory_identity"])
                # Refuse an ancestor of any other registered worktree.
                if any(
                    Path(r["path"]) != path and Path(r["path"]).is_relative_to(path)
                    for r in inventory.registrations(self.repo, self.common)
                ):
                    raise CleanupError("worktree contains another registered worktree")
                inventory.quarantine_tree(path, quarantine, row["directory_identity"])
            row["phase"] = "quarantined"
            state.save()
        # Catch agents which claimed the old cwd while the first stop ran. Once
        # renamed, new launches through the registered path cannot acquire it.
        if not replacement:
            self.stopper.stop(path, utc(now))
        if quarantine.exists():
            self.stopper.stop_moved(quarantine, path, utc(now))
        live = self.current(row)
        if live is not None:
            # Only this registration is removed; prune --expire=now would also
            # remove unrelated missing worktrees outside the agreed scope.
            if path.exists() or path.is_symlink():
                raise CleanupError("original path was recreated during expiry")
            row["head"] = live["head"]
            state.save()
        inventory.unregister(self.common, row)
        if self.current(row) is not None:
            raise CleanupError("Git registration remains after forced removal")
        row["phase"] = "unregistered"
        state.save()
        if quarantine.exists():
            inventory.remove_tree(quarantine, row["directory_identity"])
            sync_directory(quarantine.parent)
        if quarantine.exists() or quarantine.is_symlink():
            raise CleanupError("worktree directory remains after removal")
        state.log(row, decided_at=utc(now), phase="done", deleted=True)
        row.update(phase="deleted", deleted_at=utc(now))
        state.save()

    def run(
        self, *, cleanup: bool = False, dry_run: bool = False
    ) -> tuple[int, dict[str, Any]]:
        with locked(self.directory):
            state = State(self.directory, self.repo)
            now = self.clock()
            if type(now) is not int or now < 0:
                raise CleanupError("clock must return nonnegative epoch nanoseconds")
            entries = self.observe(state, now)
            results = []
            for entry in entries:
                unsafe_target = (
                    entry["expired"]
                    and entry["scope_error"]
                    and Path(entry["path"]).is_relative_to(
                        self.repo / ".claude/worktrees"
                    )
                )
                if not cleanup or dry_run or not (entry["selected"] or unsafe_target):
                    continue
                row = state.records[entry["id"]]
                result = {
                    "path": row["path"],
                    "head": row.get("head"),
                    "decided_at": utc(now),
                    "deleted": False,
                }
                try:
                    if unsafe_target:
                        raise CleanupError(entry["scope_error"])
                    self.remove(state, row, now)
                    result["deleted"] = True
                except (OSError, ValueError, CleanupError) as exc:
                    result["error"] = str(exc)
                    state.log(
                        row,
                        decided_at=utc(now),
                        phase="failed",
                        deleted=False,
                        error=str(exc),
                    )
                result["head"] = row.get("head")
                results.append(result)
            # An expired unsafe path is a failure requiring repair, not a
            # successful exemption. Scan still describes it without mutation.
            unsafe = [
                e["path"]
                for e in entries
                if e["expired"]
                and e["scope_error"]
                and Path(e["path"]).is_relative_to(self.repo / ".claude/worktrees")
            ]
            errors = (
                [f"unsafe expired worktree: {p}" for p in unsafe]
                if cleanup and not dry_run
                else []
            )
            failed = any(not r["deleted"] for r in results) or bool(errors)
            report = {
                "schema_version": 1,
                "repo": str(self.repo),
                "decided_at": utc(now),
                "mode": "cleanup" if cleanup and not dry_run else "scan",
                "dry_run": dry_run,
                "outcome": "partial_failure" if failed else "success",
                "entries": entries,
                "results": results,
                "errors": errors,
            }
            write_json(self.directory / "report-latest.json", report)
            return (1 if failed else 0), report
