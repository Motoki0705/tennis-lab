"""Fast-forward main after pinning the fetched commit and quiescing ops jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lib.checkout_lock import (
    checkout_barrier,
    checkout_lock,
    checkout_lock_holders,
    state_root,
)
from lib.windows_toast import ToastError, ToastNotSent, send_windows_toast
from ops_control.systemd import ControlError

from .git import Repository, Skip
from .runtime import Maintenance

SKIP_NOTICE_COUNT = 3


class Journal:
    def __init__(self, directory: Path, root: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = directory
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "checkout": str(root),
            "started_at": datetime.now(UTC).isoformat(),
            "outcome": "running",
            "reason": "starting",
            "head_before": None,
            "fetched_commit": None,
            "head_after": None,
            "dependency_changes": [],
            "submodule_changes": [],
        }

    def event(self, reason: str, **details: Any) -> None:
        row = {
            "at": datetime.now(UTC).isoformat(),
            "run_id": self.data["run_id"],
            "reason": reason,
            **details,
        }
        line = json.dumps(row, ensure_ascii=True, sort_keys=True)
        with (self.directory / "runs.jsonl").open("a", encoding="utf-8") as log:
            log.write(line + "\n")
        print(line, file=sys.stderr, flush=True)

    def _save_status(self, path: Path) -> None:
        temporary = self.directory / f"status-{self.data['run_id']}.tmp"
        temporary.write_text(json.dumps(self.data, ensure_ascii=True, indent=2) + "\n")
        temporary.replace(path)

    def _skip_streak(self, path: Path, now: datetime) -> dict[str, Any]:
        count, since, holders, notification = 0, now, [], "pending"
        try:
            previous = json.loads(path.read_text())
            if (
                previous["checkout"] == self.data["checkout"]
                and previous["outcome"] == "skipped"
                and (streak := previous.get("skip_streak")) is not None
            ):
                first = datetime.fromisoformat(streak["since"])
                saved_holders = streak["last_lock_holders"]
                if (
                    first.tzinfo is None
                    or type(streak["count"]) is not int
                    or streak["count"] < 1
                    or streak["notification"] not in {"pending", "sent", "uncertain"}
                    or not isinstance(saved_holders, list)
                    or any(
                        not isinstance(holder, dict)
                        or not isinstance(holder.get("entry"), str)
                        or type(holder.get("pid")) is not int
                        for holder in saved_holders
                    )
                ):
                    raise ValueError("invalid consecutive-skip record")
                count, since = streak["count"], first
                holders, notification = saved_holders, streak["notification"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.event("skip_streak_history_unavailable", error=str(exc))
        return {
            "count": count + 1,
            "since": since.isoformat(),
            "elapsed_seconds": max(0, int((now - since).total_seconds())),
            "last_lock_holders": self.data.get("lock_holders", holders),
            "notification": notification,
        }

    def _notify_skip_streak(self, streak: dict[str, Any]) -> None:
        reason = self.data["reason"]
        self.event("consecutive_skips", last_reason=reason, **streak)
        holders = []
        for holder in streak["last_lock_holders"]:
            parent = holder.get("parent_issue")
            holders.append(
                f"{holder['entry']} (PID {holder['pid']}"
                + (f", parent #{parent}" if type(parent) is int else "")
                + ")"
            )
        elapsed = timedelta(seconds=streak["elapsed_seconds"])
        try:
            send_windows_toast(
                "tennis-lab git-sync deferred",
                f"{streak['count']} consecutive skips over {elapsed}. Latest: {reason}.\n"
                + "Last checkout lock holders: "
                + (", ".join(holders) or "none observed"),
            )
        except ToastNotSent as exc:
            streak["notification"] = "pending"
            self.event("skip_streak_toast_not_sent", error=str(exc))
        except ToastError as exc:
            self.event("skip_streak_toast_uncertain", error=str(exc))
        else:
            streak["notification"] = "sent"
            self.event("skip_streak_toast_sent")

    def finish(self, outcome: str, reason: str, **details: Any) -> dict[str, Any]:
        path = self.directory / "status.json"
        # Serialize status and toast delivery even when manual runs overlap the timer.
        # Both checkout locks have already been released.
        with (self.directory / "status.lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now = datetime.now(UTC)
            self.data.update(
                outcome=outcome, reason=reason, finished_at=now.isoformat(), **details
            )
            if outcome == "skipped":
                streak = self.data["skip_streak"] = self._skip_streak(path, now)
                if (
                    streak["count"] >= SKIP_NOTICE_COUNT
                    and streak["notification"] == "pending"
                ):
                    # Persist the ambiguous outcome before a possible submission.
                    streak["notification"] = "uncertain"
                    self._save_status(path)
                    self._notify_skip_streak(streak)
            self._save_status(path)
            self.event(reason, outcome=outcome, status=self.data)
        return self.data


def run(root: Path, directory: Path) -> dict[str, Any]:
    journal = Journal(directory, root)
    repo = Repository(root)
    maintenance = Maintenance(repo.root, directory, journal.event)
    outcome, reason, details = "failed", "unexpected_error", {}
    barrier: Path | None = None

    def resume() -> None:
        if maintenance.ticket.exists() or (barrier is not None and barrier.exists()):
            try:
                current = repo.preflight()
                if barrier is not None and barrier.exists():
                    marker = json.loads(barrier.read_text())
                    if (
                        not isinstance(marker, dict)
                        or set(marker) != {"schema_version", "before", "target"}
                        or type(marker.get("schema_version")) is not int
                        or marker["schema_version"] != 1
                        or any(
                            not isinstance(marker[key], str)
                            or len(marker[key]) not in {40, 64}
                            or any(
                                char not in "0123456789abcdef" for char in marker[key]
                            )
                            for key in ("before", "target")
                        )
                    ):
                        raise ControlError("invalid checkout update marker")
                    if current not in {marker["before"], marker["target"]}:
                        raise Skip("interrupted_update_head_changed", head=current)
                    if current == marker["before"]:
                        # A failed checkout may have added untracked source
                        # files without changing any previously tracked file.
                        repo.check_target(current, marker["target"])
            except Skip as exc:
                journal.data["resume_blocked_reason"] = exc.reason
                journal.event(
                    "resident_resume_deferred",
                    checkout_reason=exc.reason,
                    **exc.details,
                )
                raise ControlError(f"resident restart deferred: {exc.reason}") from exc
            if barrier is not None and barrier.exists():
                barrier.unlink()
                journal.event("checkout_barrier_cleared")
            maintenance.resume()

    try:
        repo.validate_main()
        barrier = checkout_barrier(repo.root)
        with checkout_lock(
            repo.root, name="ops-control", exclusive=True, blocking=False
        ) as control_fd:
            if control_fd is None:
                raise Skip("control_operation_running")
            with checkout_lock(repo.root, exclusive=True, blocking=False) as code_fd:
                if code_fd is None:
                    raise Skip(
                        "checkout_jobs_running",
                        lock_holders=checkout_lock_holders(repo.root),
                    )
                maintenance.control.checkout_fd = code_fd
                # A saved resume ticket must never launch jobs from a dirty,
                # partially updated or non-main checkout after an interruption.
                head = repo.preflight()
                journal.data["head_before"] = head
                try:
                    # Complete a previous interrupted pause before a new one.
                    resume()
                    target = repo.fetch()
                    journal.data["fetched_commit"] = target
                    journal.data.update(repo.changes(head, target))
                    journal.event("fetched", snapshot=journal.data)
                    repo.check_target(head, target)
                    if head == target:
                        outcome, reason = "unchanged", "already_current"
                    else:
                        maintenance.pause()
                        # Recheck local edits and branch after network/service work.
                        if repo.preflight() != head:
                            raise Skip("head_changed_during_preflight")
                        repo.check_target(head, target)
                        with barrier.open("x", encoding="utf-8") as marker:
                            json.dump(
                                {"schema_version": 1, "before": head, "target": target},
                                marker,
                            )
                        journal.event("fast_forward", target=target)
                        repo.merge(target)
                        if repo.git("rev-parse", "HEAD^{commit}").strip() != target:
                            raise ControlError(
                                "fast-forward did not reach the pinned commit"
                            )
                        outcome, reason = "updated", "fast_forwarded"
                    journal.data["head_after"] = repo.git(
                        "rev-parse", "HEAD^{commit}"
                    ).strip()
                except Exception as exc:
                    journal.event(
                        "update_failed",
                        error=str(exc),
                        stderr=getattr(exc, "stderr", None),
                    )
                    raise
                finally:
                    try:
                        repo.release()
                    finally:
                        resume()
    except Skip as exc:
        outcome, reason, details = "skipped", exc.reason, exc.details
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ControlError,
        subprocess.SubprocessError,
    ) as exc:
        outcome, reason = "failed", "operation_failed"
        details = {"error": str(exc)}
        if isinstance(exc, subprocess.CalledProcessError):
            details["stderr"] = exc.stderr
    return journal.finish(
        outcome,
        reason,
        resume_pending=maintenance.ticket.exists(),
        checkout_blocked=bool(barrier is not None and barrier.exists()),
        **details,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)

    def interrupted(signum: int, frame: object) -> None:
        # selectors/subprocess may consume InterruptedError as a syscall EINTR.
        raise ControlError(f"git-sync interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = run(args.repo.resolve(), args.state_dir or state_root() / "git-sync")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return (
        75
        if result["outcome"] == "skipped"
        else 2
        if result["outcome"] == "failed"
        else 0
    )
