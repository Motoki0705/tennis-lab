"""Bounded common-runner discovery, with independent worktree attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from parent.manual_workers import run_directory, worker_runs
from parent.state import Store

from . import processes
from .storage import CleanupError, read_json

# These are runner directories, not workflow result.json files. Never recurse
# through arbitrary state/scratch trees (which also contain broken test data).
RUN_LOCATIONS = (
    "logs/*",
    "parent-sessions/worker-runs/*/*",
    "parent-sessions/reviews/*/agent-runtime",
    "parent-sessions/reviews/*/attempts/*/agent-runtime",
    "review/runs/*/agent-runtime",
    "dispatcher/judgements/*/agent-runtime",
    "ops-report/*/agent-runtime",
    "ops-report/*/scouts/*/runtime",
)


class RunRecords:
    def __init__(self, state_root: Path, store: Store) -> None:
        self.root = state_root
        self.store = store
        self.retired: list[Path] = []

    def directories(self) -> dict[Path, set[str]]:
        directories: dict[Path, set[str]] = {}
        self.retired = []
        for pattern in RUN_LOCATIONS:
            for directory in self.root.glob(pattern):
                if any(
                    p.is_symlink()
                    for p in (directory, *directory.parents)
                    if p != self.root.parent
                ):
                    continue
                if any(
                    (directory / name).exists()
                    for name in ("result.json", "invocation.json")
                ):
                    directories.setdefault(directory, set())
        # Explicit registered run-dir locations can live outside the state root.
        for row in self.store.read_data()["parents"].values():
            for task in row["tasks"].values():
                for run in worker_runs(task):
                    directory = run_directory(self.store, run)
                    if (
                        run["delegation_id"] != task.get("delegation_id")
                        and not directory.exists()
                        and not directory.is_symlink()
                    ):
                        # Replaced reservations can have been rejected before
                        # a local runtime was created. They are not corrupt
                        # local records. A live process reference re-adds this
                        # directory below and then requires its actual proof.
                        self.retired.append(directory)
                        continue
                    expected = directories.setdefault(directory, set())
                    if run["delegation_id"] == task.get("delegation_id"):
                        expected.add(task["worktree"])
            for job in (row.get("dispatch") or {}).get("jobs", {}).values():
                if job["phase"] == "reserved" or job.get("cancelled_before_start"):
                    continue
                if "directory" not in job and job["phase"] == "ended":
                    continue  # Historical terminal jobs can predate the runner.
                directories.setdefault(Path(job["directory"]) / "runtime", set()).add(
                    job["cwd"]
                )
        return directories

    def select(
        self,
        physical: Path,
        *,
        contract_root: Path | None = None,
        discovered_only: bool = False,
    ) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
        root = contract_root or physical
        directories = {} if discovered_only else self.directories()
        for directory in processes.run_directories(physical):
            # Tools can inherit this env from an agent running in another cwd.
            # It locates a record; only the contract/registry attributes it.
            directories.setdefault(directory, set())
        selected, errors = [], []
        diagnostics: list[dict[str, Any]] = [
            {
                "worktree": str(root),
                "directory": str(directory),
                "relation": "retired_reference",
                "reason": "superseded delegation has no local runtime directory",
            }
            for directory in self.retired
            if directory not in directories
        ]
        for directory, expected in sorted(directories.items()):
            try:
                record = read_json(directory / "result.json")
                cwd = record.get("contract", {}).get("cwd")
                if not isinstance(cwd, str) or not Path(cwd).is_absolute():
                    raise CleanupError("missing absolute runner contract.cwd")
                if any(
                    processes.within(value, root) for value in expected
                ) and not processes.within(cwd, root):
                    raise CleanupError(
                        "runner contract conflicts with registered worktree"
                    )
                if processes.within(cwd, root):
                    selected.append(directory)
            except (OSError, ValueError, CleanupError, AttributeError) as exc:
                # invocation.json is written independently of the mutable result.
                # Registry and live-process references remain authoritative even
                # if a damaged result/invocation claims a different cwd.
                evidence = set(expected)
                invocation_error = None
                try:
                    cwd = (
                        read_json(directory / "invocation.json")
                        .get("contract", {})
                        .get("cwd")
                    )
                    if not isinstance(cwd, str) or not Path(cwd).is_absolute():
                        raise CleanupError("missing absolute invocation contract.cwd")
                    evidence.add(cwd)
                except (OSError, ValueError, CleanupError, AttributeError) as other:
                    invocation_error = str(other)
                relation = (
                    "related"
                    if any(processes.within(value, root) for value in evidence)
                    else "unrelated"
                    if evidence
                    else "unknown"
                )
                reason = f"{relation} runner record unreadable: {directory}: {exc}"
                diagnostics.append(
                    {
                        "worktree": str(root),
                        "directory": str(directory),
                        "relation": relation,
                        "reason": reason,
                        "cwd_evidence": sorted(evidence),
                        "invocation_error": invocation_error,
                    }
                )
                if relation != "unrelated":
                    errors.append(reason)
        return selected, diagnostics, errors
