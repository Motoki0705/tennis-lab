"""Fast-forward main after pinning the fetched commit and quiescing ops jobs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lib.checkout_lock import checkout_lock, state_root
from ops_control.systemd import ControlError

from .git import Repository, Skip
from .runtime import Maintenance


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

    def finish(self, outcome: str, reason: str, **details: Any) -> dict[str, Any]:
        self.data.update(
            outcome=outcome,
            reason=reason,
            finished_at=datetime.now(UTC).isoformat(),
            **details,
        )
        temporary = self.directory / f"status-{self.data['run_id']}.tmp"
        temporary.write_text(json.dumps(self.data, ensure_ascii=True, indent=2) + "\n")
        temporary.replace(self.directory / "status.json")
        self.event(reason, outcome=outcome, status=self.data)
        return self.data


def run(root: Path, directory: Path) -> dict[str, Any]:
    journal = Journal(directory, root)
    repo = Repository(root)
    maintenance = Maintenance(repo.root, directory, journal.event)
    outcome, reason, details = "failed", "unexpected_error", {}
    try:
        repo.validate_main()
        with checkout_lock(
            repo.root, name="ops-control", exclusive=True, blocking=False
        ) as control_fd:
            if control_fd is None:
                raise Skip("control_operation_running")
            with checkout_lock(repo.root, exclusive=True, blocking=False) as code_fd:
                if code_fd is None:
                    raise Skip("checkout_jobs_running")
                try:
                    # Complete a previous interrupted pause before a new one.
                    maintenance.resume()
                    head = repo.preflight()
                    journal.data["head_before"] = head
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
                finally:
                    try:
                        repo.release()
                    finally:
                        maintenance.resume()
    except Skip as exc:
        outcome, reason, details = "skipped", exc.reason, exc.details
    except (OSError, ValueError, ControlError, subprocess.SubprocessError) as exc:
        outcome, reason = "failed", "operation_failed"
        details = {"error": str(exc)}
        if isinstance(exc, subprocess.CalledProcessError):
            details["stderr"] = exc.stderr
    return journal.finish(outcome, reason, **details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)

    def interrupted(signum: int, frame: object) -> None:
        raise InterruptedError(f"git-sync interrupted by signal {signum}")

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
