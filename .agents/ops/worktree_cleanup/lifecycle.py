"""Local stop coordination; no GitHub writes, ownership impersonation, or success inference."""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

from dispatcher.runtime import Runtime, alive, gate
from dispatcher.supervisor import Supervisor, managed
from parent.manual_workers import run_directory, worker_runs

from . import processes
from .registry import ExpiryStore
from .storage import CleanupError, read_json, state_root

OPS = Path(__file__).resolve().parents[1]
RUNNER = OPS / "lib/agent_exec.sh"
QUEUE = OPS.parent / "skills/training-queue/scripts/training_queue.sh"


class Stopper:
    def __init__(
        self, repo: Path, *, root: Path | None = None, timeout: float = 30.0
    ) -> None:
        self.repo = repo
        self.root = root or state_root()
        self.store = ExpiryStore(self.root / "parent-sessions")
        self.timeout = timeout

    def stop_run(
        self,
        directory: Path,
        root: Path,
        runner: Path = RUNNER,
        *,
        task_id: str | None = None,
        branch: str | None = None,
    ) -> None:
        record = read_json(directory / "result.json")
        self.check_contract(record, root, task_id, branch)
        runtime = Runtime(runner)
        proof = runtime.inspect(directory, stop=True)
        deadline = time.monotonic() + self.timeout
        while not runtime.ended(proof):
            if time.monotonic() >= deadline:
                raise CleanupError(
                    f"runner termination/cleanup unconfirmed: {directory}"
                )
            Event().wait(0.05)
            proof = runtime.inspect(directory)
        self.check_contract(proof["record"], root, task_id, branch)

    @staticmethod
    def check_contract(
        record: dict[str, Any], root: Path, task_id: str | None, branch: str | None
    ) -> None:
        contract = record.get("contract")
        if (
            not isinstance(contract, dict)
            or not processes.within(contract.get("cwd", ""), root)
            or (task_id is not None and contract.get("task_id") != task_id)
            or (branch is not None and contract.get("branch") != branch)
        ):
            raise CleanupError(
                "runner contract does not match the expired worktree/delegation"
            )

    def registry(self, root: Path, native: list[dict[str, Any]], stamp: str) -> None:
        if not self.store.path.exists():
            return
        sessions = {session for agent in native for session in agent["sessions"]}
        # Existing manual parents predate owner_runtime. Herdr already pins the
        # session's native process by PID/start time in its private pane record.
        owners = set()
        for path in (self.root / "herdr-labels/panes").glob("*.json"):
            pane = read_json(path)
            if pane.get("version") != 1 or pane.get("state") != "active":
                continue
            if any(
                stamp.get("pid") == agent["identity"]["pid"]
                and stamp.get("start") == agent["identity"]["start_ticks"]
                for stamp in pane.get("processes", [])
                for agent in native
            ):
                owners.add(pane.get("owner"))
        plans = []
        with self.store.locked():
            for key, row in self.store.parents.items():
                dispatch = managed(row)
                saved_runtime = row.get("owner_runtime", {})
                saved_owner = processes.within(saved_runtime.get("cwd", ""), root)
                if saved_owner and saved_runtime.get("process") is not None:
                    native.append(
                        {
                            "identity": saved_runtime["process"],
                            "sessions": [row["session"]],
                        }
                    )
                owner = (
                    saved_owner
                    or row["session"] in sessions
                    or hashlib.sha256(row["session"].encode()).hexdigest() in owners
                    or bool(
                        dispatch
                        and any(
                            job["role"] == "parent"
                            and processes.within(job["cwd"], root)
                            for job in dispatch["jobs"].values()
                        )
                    )
                )
                selected = [
                    n
                    for n, task in row["tasks"].items()
                    if processes.within(task.get("worktree", ""), root)
                ]
                if owner:
                    selected = list(row["tasks"])
                    row["worktree_expiry"] = {
                        "path": str(root),
                        "decided_at": stamp,
                        "result": "stop_requested",
                    }
                    # This is forced release, never successful completion.
                    row["status"] = "released"
                    if dispatch:
                        dispatch.update(
                            phase="stopped",
                            reason="worktree_expired",
                            wake_requested=False,
                        )
                for n in selected:
                    row["tasks"][n].update(
                        stopped=True,
                        worktree_expiry={
                            "path": str(root),
                            "decided_at": stamp,
                            "result": "stop_requested",
                        },
                    )
                if selected or owner:
                    plans.append((key, copy.deepcopy(row), selected, owner))
            self.store.save()
        errors = []
        for key, row, selected, owner in plans:
            failed = False
            try:
                dispatch = managed(row)
                if dispatch:
                    for job in dispatch["jobs"].values():
                        if not owner and str(job["child"]) not in selected:
                            continue
                        if job["phase"] == "ended":
                            continue
                        if job["phase"] == "reserved":
                            Supervisor.cancel_reservation(row, job, "worktree_expired")
                            if job["phase"] != "ended":
                                raise CleanupError(
                                    "reserved job has unconfirmed launch evidence"
                                )
                        else:
                            directory = Path(job["directory"])
                            path = directory / "runtime"
                            self.stop_run(
                                path,
                                Path(job["cwd"]),
                                Path(job["runner"]),
                                task_id=job["task_id"],
                            )
                            with gate(directory / "lifetime.lock") as fd:
                                if fd is None or alive(job.get("worker")):
                                    raise CleanupError(
                                        "dispatcher launcher termination unconfirmed"
                                    )
                            job.update(phase="ended", diagnostic="worktree_expired")
                for n in selected:
                    task = row["tasks"][n]
                    if not task.get("delegated"):
                        continue
                    if not dispatch:
                        runs = [
                            r
                            for r in worker_runs(task)
                            if r["delegation_id"] == task.get("delegation_id")
                        ]
                        if (
                            not runs
                            and "worker_ended" not in task
                            and "worker_runs" not in task
                        ):
                            raise CleanupError(
                                "delegated worker has no runner termination evidence"
                            )
                        for run in runs:
                            self.stop_run(
                                run_directory(self.store, run),
                                Path(task["worktree"]),
                                task_id=task["delegation_id"],
                                branch=task.get("branch"),
                            )
                    task["worker_ended"] = {
                        "delegation_id": task.get("delegation_id", "legacy"),
                        "evidence": "worktree expiry: common runner confirmed termination and cleanup, or cancelled before launch",
                    }
                    task["worktree_expiry"]["result"] = "stopped"
            except (OSError, ValueError, CleanupError) as exc:
                failed = True
                errors.append(str(exc))
                row["worktree_expiry"] = {
                    "path": str(root),
                    "decided_at": stamp,
                    "result": "stop_failed",
                    "error": str(exc),
                }
                if row.get("dispatch"):
                    row["dispatch"].update(
                        phase="recovery_required", reason="worktree_expiry_stop_failed"
                    )
            with self.store.locked():
                current = self.store.parents[key]
                if (
                    current["session"] != row["session"]
                    or current["generation"] != row["generation"]
                ):
                    raise CleanupError("parent ownership changed during expiry stop")
                for n in selected:
                    if current["tasks"][n].get("delegation_id") != row["tasks"][n].get(
                        "delegation_id"
                    ):
                        raise CleanupError(
                            "worker delegation changed during expiry stop"
                        )
                    for field in ("worker_ended", "worktree_expiry", "preflight"):
                        if field in row["tasks"][n]:
                            current["tasks"][n][field] = row["tasks"][n][field]
                if row.get("dispatch"):
                    # Jobs may finish concurrently; do not replace an entire old row.
                    for job_id, job in row["dispatch"]["jobs"].items():
                        if job["phase"] == "ended":
                            current["dispatch"]["jobs"][job_id].update(job)
                    if failed:
                        current["dispatch"].update(
                            phase="recovery_required",
                            reason="worktree_expiry_stop_failed",
                        )
                if "worktree_expiry" in row:
                    current["worktree_expiry"] = row["worktree_expiry"]
                self.store.save()
        if errors:
            raise CleanupError("; ".join(errors))

    def queue(self, root: Path) -> None:
        # Reuse the existing shell-word decoder, including printf %q paths.
        sys.path.insert(0, str(OPS / "cleanup"))
        legacy = importlib.import_module("inventory")

        queue = Path(
            os.environ.get("TRAINING_QUEUE_DIR") or self.repo / ".training_queue"
        )
        selected = set()
        for state in ("jobs", "running"):
            directory = queue / state
            if not directory.exists():
                continue
            for path in directory.glob("*.job"):
                try:
                    text = path.read_text()
                except FileNotFoundError:
                    continue
                words = tuple(legacy.shell_words(text))
                paths, cwds = legacy.job_paths(words)
                job = legacy.QueueJob(
                    path.name, text, words, paths, cwds, legacy.nested_code(text, words)
                )
                if legacy.queue_refs_for([job], str(root)):
                    selected.add(path.name)
        for name in sorted(selected):
            result = subprocess.run(
                ["bash", str(QUEUE), "cancel", name],
                cwd=self.repo,
                env={**os.environ, "TRAINING_QUEUE_DIR": str(queue)},
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise CleanupError(
                    f"queue cancellation failed: {name}: {result.stderr.strip()}"
                )
            deadline = time.monotonic() + self.timeout
            while True:
                sys.path.insert(0, str(OPS.parents[1]))
                history = importlib.import_module("src.utils.training_queue")
                terminal = history.read_completion(queue, name) or {}
                if terminal.get("state") in {"cancelled", "done", "failed"} and not any(
                    (queue / state / name).exists() for state in ("jobs", "running")
                ):
                    break
                if time.monotonic() >= deadline:
                    raise CleanupError(f"queue termination unconfirmed: {name}")
                Event().wait(0.05)

    def loose_runs(
        self,
        root: Path,
        *,
        contract_root: Path | None = None,
        discovered_only: bool = False,
    ) -> None:
        directories = set()
        for directory, subdirs, files in (
            os.walk(self.root, followlinks=False) if not discovered_only else ()
        ):
            subdirs[:] = [
                name for name in subdirs if not (Path(directory) / name).is_symlink()
            ]
            if "result.json" not in files:
                continue
            directories.add(Path(directory))
        # Explicit --run-dir locations need not be below the default state root.
        directories.update(processes.run_directories(root))
        for runtime_directory in directories:
            path = runtime_directory / "result.json"
            record = read_json(path)
            contract = record.get("contract")
            if isinstance(contract, dict) and processes.within(
                contract.get("cwd", ""), contract_root or root
            ):
                self.stop_run(path.parent, contract_root or root)

    def stop_moved(self, root: Path, original: Path, stamp: str) -> None:
        # Only processes physically in quarantine are selected. Replaying an
        # old deletion must not stop a replacement's jobs at the original path.
        self.loose_runs(root, contract_root=original, discovered_only=True)
        for agent in processes.agents(root):
            processes.stop_tree(agent["identity"])
        if processes.agents(root):
            raise CleanupError("native agents remain in quarantined worktree")

    def stop(self, root: Path, stamp: str) -> None:
        native = processes.agents(root)
        errors = []
        for action in (
            lambda: self.registry(root, native, stamp),
            lambda: self.queue(root),
            lambda: self.loose_runs(root),
        ):
            try:
                action()
            except (OSError, ValueError, CleanupError) as exc:
                errors.append(str(exc))
        for agent in native:
            try:
                processes.stop_tree(agent["identity"])
            except (OSError, ValueError, CleanupError) as exc:
                errors.append(str(exc))
        if processes.agents(root):
            errors.append("native agents remain in expired worktree")
        if self.store.path.exists():
            with self.store.locked():
                for row in self.store.parents.values():
                    expiry = row.get("worktree_expiry", {})
                    if expiry.get("path") == str(root):
                        expiry.update(
                            result="stop_failed" if errors else "stopped", errors=errors
                        )
                self.store.save()
        if errors:
            raise CleanupError("; ".join(errors))
