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
from dispatcher.supervisor import Supervisor, event, managed
from parent.manual_workers import run_directory, worker_runs

from . import processes
from .registry import ExpiryStore
from .run_records import RunRecords
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
        self.diagnostics: list[dict[str, Any]] = []

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
                saved_owner = (
                    saved_runtime.get("session") == row["session"]
                    and saved_runtime.get("generation") == row["generation"]
                    and processes.within(saved_runtime.get("cwd", ""), root)
                )
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
                            and dispatch.get("session") == row["session"]
                            and job.get("authorization", {}).get("generation")
                            == row["generation"]
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
                            "delegation_id": row["tasks"][n].get("delegation_id"),
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
                            deadline = time.monotonic() + self.timeout
                            while True:
                                with gate(directory / "lifetime.lock") as fd:
                                    finished = fd is not None and not alive(
                                        job.get("worker")
                                    )
                                if finished:
                                    break
                                if time.monotonic() >= deadline:
                                    raise CleanupError(
                                        "dispatcher launcher termination unconfirmed"
                                    )
                                Event().wait(0.05)
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
                        if not runs and "worker_runs" not in task:
                            # This predates common-runner provenance. Do not
                            # invent runner cleanup evidence or end it before
                            # native descendants have actually been confirmed.
                            task["worktree_expiry"]["result"] = (
                                "awaiting_process_confirmation"
                            )
                            continue
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
                            target = current["dispatch"]["jobs"][job_id]
                            if (
                                job.get("cancelled_before_start")
                                and target["phase"] != "ended"
                            ):
                                event(
                                    current["dispatch"],
                                    "runtime_cancelled",
                                    job_id=job_id,
                                    reason="worktree_expired",
                                )
                            # Preserve exit/launcher evidence written while stop
                            # waited outside the registry lock.
                            for field in (
                                "phase",
                                "diagnostic",
                                "cancelled_before_start",
                            ):
                                if field in job:
                                    target[field] = job[field]
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

    def finish_legacy(self, root: Path, stamp: str, *, physical: Path) -> None:
        """Certify local absence after all managed and native stops succeeded."""
        if not self.store.path.exists():
            return
        view = processes.snapshot()
        remaining, _ = processes.scoped_processes(physical, view)
        if remaining:
            raise CleanupError(
                f"local processes remain before legacy termination: {remaining}"
            )
        with self.store.locked():
            for row in self.store.parents.values():
                for task in row["tasks"].values():
                    expiry = task.get("worktree_expiry", {})
                    if (
                        expiry.get("path") != str(root)
                        or expiry.get("result") != "awaiting_process_confirmation"
                        or not task.get("stopped")
                        or not task.get("delegated")
                        or expiry.get("delegation_id") != task.get("delegation_id")
                    ):
                        continue
                    # An expiring owner also stops tasks in other worktrees.
                    # A legacy child outside this root cannot be ended merely
                    # because the owner's cwd has become empty.
                    task_root = Path(task["worktree"])
                    if processes.within(str(task_root), root):
                        task_root = physical / task_root.relative_to(root)
                    remaining, _ = processes.scoped_processes(task_root, view)
                    if remaining:
                        raise CleanupError(
                            f"legacy child still has local processes: {remaining}"
                        )
                    # No shared-runner result exists for this old delegation.
                    # This is a forced local termination, never task success or
                    # a claim about a remote machine's agent session.
                    task["worker_ended"] = {
                        "delegation_id": task.get("delegation_id", "legacy"),
                        "evidence": "worktree expiry: legacy delegation without runner records; local cwd processes and captured native descendants confirmed absent",
                    }
                    expiry.update(
                        result="stopped",
                        termination={
                            "source": "local_process_absence",
                            "checked_at": stamp,
                            "path": str(root),
                            "verified_cwd": str(task_root),
                            "runner_record": None,
                        },
                    )
            self.store.save()

    def preflight(self, root: Path) -> dict[str, Any]:
        """Read-only stop planning; never invoke runner status/stop or repair."""
        runs, diagnostics, errors = RunRecords(self.root, self.store).select(root)
        view = processes.snapshot()
        scoped, _ = processes.scoped_processes(root, view)
        scoped.extend(agent["identity"] for agent in processes.agents(root))
        saved = processes.read_stop_proof(self.native_proof(root))
        recovery = processes.unscoped_recovery_evidence(view, scoped, saved, root)
        if saved is not None and recovery is None:
            if not saved.get("capture_complete", True):
                errors.append(
                    "native discovery was interrupted; original roots cannot be certified"
                )
            tracked = {processes.identity_key(item): item for item in saved["tracked"]}
            unconfirmed = {
                processes.identity_key(item): item
                for item in saved.get("unconfirmed", [])
            }
            processes.expand_tracked(tracked, view)
            processes.expand_tracked(unconfirmed, view)
            unknown = processes.uncertain_processes(
                view,
                tracked,
                {processes.identity_key(item): item for item in saved["baseline"]},
                {processes.identity_key(item): item for item in saved["ancestors"]},
                unconfirmed,
            )
            if unknown:
                errors.append(f"unconfirmed native process provenance: {unknown}")
        legacy = [
            n
            for row in self.store.read_data()["parents"].values()
            for n, task in row["tasks"].items()
            if task.get("delegated")
            and "worker_runs" not in task
            and processes.within(task.get("worktree", ""), root)
        ]
        return {
            "path": str(root),
            "runner_runs": len(runs),
            "native_scope": scoped,
            "legacy_delegations": legacy,
            "legacy_proof_recovery": recovery,
            "diagnostics": diagnostics,
            "errors": errors,
        }

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
        directories, diagnostics, errors = RunRecords(self.root, self.store).select(
            root, contract_root=contract_root, discovered_only=discovered_only
        )
        self.diagnostics.extend(diagnostics)
        for directory in directories:
            try:
                self.stop_run(directory, contract_root or root)
            except (OSError, ValueError, CleanupError) as exc:
                errors.append(f"related runner stop failed: {directory}: {exc}")
        if errors:
            raise CleanupError("; ".join(errors))

    def stop_moved(self, root: Path, original: Path, stamp: str) -> None:
        # Only processes physically in quarantine are selected. Replaying an
        # old deletion must not stop a replacement's jobs at the original path.
        view, native = self.prepare_native(root)
        self.loose_runs(root, contract_root=original, discovered_only=True)
        self.stop_native(root, native, view)
        if processes.agents(root):
            raise CleanupError("native agents remain in quarantined worktree")

    def native_proof(self, root: Path) -> Path:
        return (
            self.root
            / "native-stops"
            / (hashlib.sha256(os.fsencode(root)).hexdigest() + ".json")
        )

    def prepare_native(self, root: Path) -> tuple[dict[int, Any], list[dict[str, Any]]]:
        # Take the baseline before even enumerating agents. Retain it across
        # registry/queue/runner calls which can make a parent exit and orphan a child.
        view = processes.snapshot()
        proof = self.native_proof(root)
        fresh = processes.begin_stop(view, proof)
        native = processes.agents(root)
        seeds, excluded = processes.scoped_processes(root, view)
        seeds.extend(agent["identity"] for agent in native)
        recovery = processes.recover_unscoped_stop(view, seeds, proof, root)
        if recovery is not None:
            self.diagnostics.append(recovery)
        after = processes.snapshot()
        lost = [
            node["identity"]
            for pid, node in view.items()
            if after.get(pid, {}).get("identity") != node["identity"]
            and processes.identity_key(node["identity"]) not in excluded
        ]
        processes.prepare_stop(
            view, seeds, proof, capture_completed=fresh, uncertain_roots=lost
        )
        return view, native

    def stop_native(
        self, root: Path, native: list[dict[str, Any]], view: dict[int, Any]
    ) -> None:
        proof = self.native_proof(root)
        # registry() may add a saved owner identity. Resolve it against the
        # original ancestry, not a snapshot taken after stopping that owner.
        processes.prepare_stop(view, [agent["identity"] for agent in native], proof)
        if native:
            for agent in native:
                processes.stop_tree(agent["identity"], proof=proof)
        else:
            processes.stop_tree(None, proof=proof)

    def stop(self, root: Path, stamp: str) -> None:
        view, native = self.prepare_native(root)
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
        try:
            self.stop_native(root, native, view)
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
