"""Durable reservations and runtime events in the existing parent registry."""

from __future__ import annotations

import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from parent.adopted_tasks import check_task_adoptions
from parent.execution import (
    authorization,
    requirement_body,
    validate_authorization,
    validate_candidate,
)
from parent.github import GitHub
from parent.policy import closing_issues, digest, nonempty, number
from parent.session import ParentSession
from parent.state import Store, session_identity

from . import judgement
from .collect import assert_material_fresh, collect, material, source_references
from .outcomes import read_result
from .runtime import Runtime, alive, gate, load, process_identity
from .schedule import Schedule, latest_slot, next_slot
from .state import DispatchStore

PHASES = {
    "reserved",
    "running",
    "waiting",
    "resume_pending",
    "failed",
    "stopped",
    "completed",
    "recovery_required",
}
JOB_PHASES = {"reserved", "starting", "ended", "recovery_required"}


def managed(row: dict[str, Any]) -> dict[str, Any] | None:
    if "dispatch" not in row:
        return None
    value = row["dispatch"]
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("phase") not in PHASES
        or not isinstance(value.get("jobs"), dict)
        or not isinstance(value.get("events"), list)
        or type(value.get("child_limit")) is not int
        or value["child_limit"] < 1
        or not isinstance(value.get("assignment_id"), str)
        or not value["assignment_id"]
        or not isinstance(value.get("session"), str)
        or value["session"] != row["session"]
    ):
        raise ValueError(
            "malformed dispatcher assignment or owner changed; reconcile explicitly"
        )
    validate_authorization(value.get("authorization"))
    if value["authorization"]["generation"] != row["generation"]:
        raise ValueError("dispatcher owner generation changed")
    if value.get("active_job") not in value["jobs"]:
        raise ValueError("active parent runtime is missing")
    for key, job in value["jobs"].items():
        if (
            not isinstance(job, dict)
            or job.get("id") != key
            or job.get("phase") not in JOB_PHASES
            or job.get("role") not in {"parent", "implementation"}
            or not all(
                isinstance(job.get(k), str) and job[k]
                for k in ("directory", "cwd", "runner", "task_id")
            )
        ):
            raise ValueError("invalid managed runtime record")
    return value


def children_pending(row: dict[str, Any]) -> list[int]:
    # Reuse the reservation/end-evidence contract from parent.launch. A stopped
    # or closed task still counts until the current delegation has ended.
    pending = set()
    for key, task in row["tasks"].items():
        if not task.get("delegated"):
            continue
        ended = task.get("worker_ended")
        if ended is None:
            pending.add(number(int(key)))
        elif (
            not isinstance(ended, dict)
            or set(ended) != {"delegation_id", "evidence"}
            or ended["delegation_id"] != task.get("delegation_id", "legacy")
        ):
            raise ValueError("stale or malformed child end evidence")
        else:
            nonempty(ended["evidence"], "worker-end evidence")
    dispatch = managed(row)
    if dispatch:
        # Keep old runtime records as well: replacing a delegation must never
        # erase a live worker from the capacity calculation.
        pending.update(
            job["child"]
            for job in dispatch["jobs"].values()
            if job["role"] == "implementation" and job["phase"] != "ended"
        )
    return sorted(pending)


def occupies(row: dict[str, Any]) -> bool:
    value = managed(row)
    return bool(
        value
        and (
            children_pending(row)
            or any(j["phase"] != "ended" for j in value["jobs"].values())
            or value["phase"] in {"reserved", "running", "recovery_required"}
        )
    )


def event(dispatch: dict[str, Any], kind: str, **detail: Any) -> None:
    dispatch["events"].append(
        {"sequence": len(dispatch["events"]) + 1, "event": kind, **detail}
    )


def unstarted(job: dict[str, Any]) -> bool:
    directory = Path(job["directory"])
    return bool(
        job["worker"] is None
        and job["runner_exit"] is None
        and not any(
            (directory / name).exists() or (directory / name).is_symlink()
            for name in ("runtime", "runner.stdout", "runner.stderr")
        )
    )


def released_reservation(row: dict[str, Any]) -> bool:
    dispatch = managed(row)
    return bool(
        dispatch
        and row["status"] == "released"
        and row["session"] == f"dispatcher-reservation:{dispatch['assignment_id']}"
        and not row["tasks"]
        and all(
            j["phase"] == "ended" and j.get("cancelled_before_start") and unstarted(j)
            for j in dispatch["jobs"].values()
        )
    )


class Supervisor:
    def __init__(self, gh: GitHub, store: Store, schedule: Schedule) -> None:
        if (
            isinstance(store, DispatchStore)
            and store.schedule.directory != schedule.directory
        ):
            raise ValueError("dispatcher store and schedule differ")
        self.gh, self.store, self.schedule = (
            gh,
            store
            if isinstance(store, DispatchStore)
            else DispatchStore(store.directory, schedule),
            schedule,
        )

    def _session(self, key: str, row: dict[str, Any]) -> ParentSession:
        if key.rpartition("#")[0] != self.gh.repo.casefold():
            raise ValueError("managed repository mismatch")
        return ParentSession(
            self.gh, self.store, number(int(key.rpartition("#")[2])), row["session"]
        )

    def _authorized(self, key: str, row: dict[str, Any]) -> None:
        dispatch = managed(row)
        if (
            not dispatch
            or authorization(self._session(key, row)) != dispatch["authorization"]
        ):
            raise ValueError("managed execution basis changed")

    def _validate_completion_resume(
        self,
        session: ParentSession,
        row: dict[str, Any],
        job: dict[str, Any],
        issue: dict[str, Any],
    ) -> None:
        """Resume an accepted standalone parent only to verify its merged work."""
        if issue.get("state_reason") != "completed" or job["resume_from"] is None:
            raise ValueError(
                "closed parent requires completed merge evidence and an accepted resume source"
            )
        _, items = session.work_items()
        if set(items) != {session.parent}:
            raise ValueError(
                "closed parent recovery requires the standalone implementation target"
            )
        parents = [
            previous
            for previous in row["dispatch"]["jobs"].values()
            if previous["role"] == "parent" and previous["sequence"] < job["sequence"]
        ]
        if not parents:
            raise ValueError("closed parent has no accepted parent runtime")
        previous_parent = max(parents, key=lambda j: j["sequence"])
        if (
            previous_parent["authorization"] != job["authorization"]
            or previous_parent["assignment_id"] != job["assignment_id"]
            or self.resume_source(row, previous_parent) != job["resume_from"]
        ):
            raise ValueError(
                "closed parent resume source differs from the accepted session"
            )
        implementations = [
            previous
            for previous in row["dispatch"]["jobs"].values()
            if previous["role"] == "implementation"
            and previous["child"] == session.parent
        ]
        task = row["tasks"].get(str(session.parent), {})
        if not implementations:
            raise ValueError(
                "closed parent has no implementation delegation to reconcile"
            )
        previous = max(implementations, key=lambda j: j["sequence"])
        if (
            not task.get("delegated")
            or not task.get("delegation_id")
            or task["delegation_id"] != previous["delegation_id"]
            or task.get("authorization") != job["authorization"]
            or previous["authorization"] != job["authorization"]
            or previous["assignment_id"] != job["assignment_id"]
            or task.get("worktree") != previous["cwd"]
            or task.get("branch") != previous["branch"]
        ):
            raise ValueError(
                "closed parent implementation delegation/generation changed"
            )
        from parent.development import verify_completion

        verify_completion(session)

    def _validate_start(
        self, key: str, row: dict[str, Any], job: dict[str, Any]
    ) -> None:
        if not unstarted(job):
            raise ValueError("reserved runtime already has launch evidence")
        dispatch = managed(row)
        if (
            not dispatch
            or dispatch["phase"] in {"stopped", "recovery_required"}
            or job["authorization"] != dispatch["authorization"]
            or job["assignment_id"] != dispatch["assignment_id"]
            or job["candidate_sha256"] != digest(dispatch["candidate"])
        ):
            raise ValueError(
                "runtime assignment/generation/authorization changed or stopped"
            )
        self._authorized(key, row)
        session = self._session(key, row)
        issue = self.gh.issue(session.parent)
        if job["role"] == "parent" and issue["state"] == "closed":
            self._validate_completion_resume(session, row, job, issue)
        else:
            session.agreed()
        check_task_adoptions(self.gh, self.gh.issue(session.parent)["body"])
        if job["role"] == "parent":
            if dispatch["active_job"] != job["id"]:
                raise ValueError("parent reservation was superseded")
        else:
            task = row["tasks"].get(str(job["child"]), {})
            if (
                not task.get("delegated")
                or task.get("delegation_id") != job["delegation_id"]
                or task.get("worker_ended")
                or task.get("stopped")
                or task.get("authorization") != job["authorization"]
                or task.get("agent") != "codex"
                or task.get("worktree") != job["cwd"]
                or task.get("branch") != job["branch"]
            ):
                raise ValueError("current child delegation changed or stopped")
            _, children = session.work_items()
            child = children.get(job["child"])
            if child is None or task.get("requirements_sha256") != digest(
                requirement_body(child.body)
            ):
                raise ValueError(
                    "delegated child requirements or formal parent changed"
                )
            check_task_adoptions(self.gh, child.body)
            planned = next(
                t
                for t in session.launch_plan(restart_child=job["child"])["tasks"]
                if t["child"] == job["child"]
            )
            reasons = planned["reasons"]
            if job["resume_from"] is not None and task.get("restart_evidence"):
                related = [
                    p
                    for p in self.gh.pulls()
                    if job["child"] in closing_issues(p.get("body") or "", self.gh.repo)
                ]
                if related and (
                    len(related) != 1
                    or related[0]["state"] != "open"
                    or related[0]["head"]["ref"] != job["branch"]
                    or related[0]["head"]["repo"]["full_name"].casefold()
                    != self.gh.repo.casefold()
                ):
                    raise ValueError(
                        "continuation requires the same open implementation PR"
                    )
                reasons = [r for r in reasons if r["code"] != "existing_pr"]
            if reasons:
                raise ValueError(
                    "child launch is blocked: "
                    + "; ".join(r["message"] for r in reasons)
                )
        snapshot = load(Path(job["directory"]) / "start-input.json")
        sources = dispatch["candidate"]["sources"]
        if (
            digest(material(snapshot, session.parent, sources))
            != job["start_input_sha256"]
        ):
            raise ValueError("reserved runtime input digest changed")
        cited_issues, _, _ = source_references(self.gh.repo.casefold(), sources)
        current = collect(
            self.gh,
            self.store,
            Path(self.schedule.read()["config"]["cwd"]),
            [session.parent, *cited_issues],
        )
        # The dispatcher itself changes this row while binding/finishing jobs.
        # Its authorization and the current delegation were checked above;
        # all external inputs and other owners remain part of freshness.
        current["owners"].pop(str(session.parent), None)
        if str(session.parent) in snapshot["owners"]:
            current["owners"][str(session.parent)] = snapshot["owners"][
                str(session.parent)
            ]
        assert_material_fresh(snapshot, current, session.parent, sources)

    def claim(self, key: str, job_id: str) -> dict[str, Any] | None:
        """Validate and acquire a reservation in one parent registry lock."""
        with self.store.locked():
            row = self.store.parents[key]
            dispatch = row["dispatch"]
            job: dict[str, Any] = dispatch["jobs"][job_id]
            if job["phase"] != "reserved" or not self.schedule.read()["enabled"]:
                return None
            try:
                self._validate_start(key, row, job)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                reason = f"{exc}; 次の予定回または明示実行で再判断する"
                self.cancel_reservation(row, job, reason)
                if (
                    job["role"] == "parent"
                    and job["phase"] == "ended"
                    and dispatch["phase"] != "stopped"
                ):
                    dispatch.update(phase="failed", reason=reason)
                    if (
                        row["session"]
                        == f"dispatcher-reservation:{job['assignment_id']}"
                        and row["generation"] == job["authorization"]["generation"]
                    ):
                        row["status"] = "released"
                self.store.save()
                return None
            job.update(phase="starting", worker=process_identity(os.getpid()))
            event(dispatch, "runtime_starting", job_id=job_id)
            self.store.save()
            return job

    def resume_source(
        self, row: dict[str, Any], previous: dict[str, Any]
    ) -> str | None:
        jobs = row["dispatch"]["jobs"]
        while True:
            if previous["phase"] != "ended":
                raise ValueError("reconcile the previous runtime before resuming")
            if not previous.get("cancelled_before_start"):
                directory = Path(previous["directory"]) / "runtime"
                record = load(directory / "result.json")
                session = record["runtime"].get("session_id")
                if not session or (
                    previous["role"] == "parent"
                    and row["session"] != f"claude:{session}"
                ):
                    raise ValueError(
                        "previous runtime has no matching resumable session"
                    )
                return str(directory)
            if not unstarted(previous):
                raise ValueError(
                    "cancelled runtime has launch evidence; reconcile explicitly before resuming"
                )
            source = previous["resume_from"]
            if source is None:
                return None
            matches = [
                j
                for j in jobs.values()
                if str(Path(j["directory"]) / "runtime") == source
                and j["role"] == previous["role"]
                and j["child"] == previous["child"]
                and j["sequence"] < previous["sequence"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    "cancelled reservation has no unique saved resume source"
                )
            previous = matches[0]

    @staticmethod
    def cancel_reservation(
        row: dict[str, Any], job: dict[str, Any], reason: str
    ) -> None:
        dispatch = row["dispatch"]
        if not unstarted(job):
            job.update(
                phase="recovery_required",
                diagnostic="予約に起動証拠があるため実プロセスを照合する",
            )
            dispatch.update(phase="recovery_required", reason=job["diagnostic"])
            return
        job.update(phase="ended", cancelled_before_start=True, diagnostic=reason)
        event(dispatch, "runtime_cancelled", job_id=job["id"], reason=reason)
        task = row["tasks"].get(str(job["child"]), {})
        if (
            job["child"] is not None
            and task.get("delegation_id") == job["delegation_id"]
        ):
            task["worker_ended"] = {
                "delegation_id": job["delegation_id"],
                "evidence": "dispatcher cancelled reservation before the worker could claim/start the runner: "
                + reason,
            }

    def _job(
        self,
        key: str,
        row: dict[str, Any],
        *,
        child: int | None = None,
        delegation_id: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        dispatch = row["dispatch"]
        config = self.schedule.read()["config"]
        task = row["tasks"][str(child)] if child is not None else {}
        cwd = (
            config["cwd"]
            if child is None
            else nonempty(task.get("worktree"), "delegated worktree")
        )
        branch = (
            None if child is None else nonempty(task.get("branch"), "delegated branch")
        )
        if not Path(cwd).is_absolute():
            raise ValueError("delegated worktree must be absolute")
        parent = number(int(key.rpartition("#")[2]))
        sources = dispatch["candidate"]["sources"]
        if row["session"].startswith("dispatcher-reservation:") and child is None:
            run = next(
                r
                for r in self.schedule.read()["runs"]
                if r["id"] == dispatch["source_run"]
            )
            snapshot = load(Path(run["directory"]) / "input.json")
        else:
            cited_issues, _, _ = source_references(self.gh.repo.casefold(), sources)
            snapshot = collect(
                self.gh, self.store, Path(config["cwd"]), [parent, *cited_issues]
            )
        job_id = uuid.uuid4().hex
        directory = (
            self.schedule.directory / "workers" / dispatch["assignment_id"] / job_id
        )
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        job = {
            "id": job_id,
            "sequence": len(dispatch["jobs"]) + 1,
            "phase": "reserved",
            "role": "implementation" if child is not None else "parent",
            "child": child,
            "delegation_id": delegation_id,
            "directory": str(directory),
            "runner": config["runner"],
            "dispatcher_state_dir": str(self.schedule.directory),
            "cwd": cwd,
            "branch": branch,
            "task_id": f"development:{key}:generation:{row['generation']}"
            + (f":child:{child}" if child is not None else ""),
            "model": model,
            "resume_from": resume_from,
            "worker": None,
            "runner_exit": None,
            "diagnostic": None,
            "authorization": dict(dispatch["authorization"]),
            "assignment_id": dispatch["assignment_id"],
            "candidate_sha256": digest(dispatch["candidate"]),
            "start_input_sha256": digest(material(snapshot, parent, sources)),
        }
        Store.write_json(directory / "start-input.json", snapshot)
        if child is None:
            inputs = {
                "schema_version": 1,
                "repo": self.gh.repo.casefold(),
                "parent": number(int(key.rpartition("#")[2])),
                "assignment_id": dispatch["assignment_id"],
                "job_id": job_id,
                "authorization": dispatch["authorization"],
                "candidate": dispatch["candidate"],
                "parent_state_dir": str(self.store.directory.resolve()),
                "dispatcher_state_dir": str(self.schedule.directory),
                "runtime_dir": str(directory / "runtime"),
                "max_implementation_children": dispatch["child_limit"],
            }
            Store.write_json(directory / "input.json", inputs)
            prompt = (
                Path(__file__)
                .with_name("parent_bootstrap.md")
                .read_text(encoding="utf-8")
            )
            prompt += f"\n\nbootstrap_input: {directory / 'input.json'}\n"
        (directory / "prompt.md").write_text(
            nonempty(prompt, "runtime prompt"), encoding="utf-8"
        )
        dispatch["jobs"][job_id] = job
        if child is None:
            dispatch.update(active_job=job_id, phase="reserved", wake_requested=False)
        event(
            dispatch,
            "runtime_reserved",
            job_id=job_id,
            child=child,
            resume_from=resume_from,
        )
        return job

    def assign(self) -> list[int]:
        """No model invocation. Consume only committed candidates, in priority order."""
        assigned: list[int] = []
        with self.store.locked():
            state = self.schedule.read()
            if not state["enabled"]:
                return assigned
            for run in state["runs"]:
                if run["phase"] != "recorded":
                    continue
                directory = Path(run["directory"])
                result = read_result(directory)
                if result["result"] != "recorded":
                    raise ValueError("candidate journal differs from recorded judgment")
                snapshot = load(directory / "input.json")
                for candidate in sorted(
                    result["candidates"], key=lambda c: c["priority"]
                ):
                    n = number(candidate["parent"])
                    status = run["candidates"].get(str(n))
                    if status and status["state"] != "capacity_wait":
                        continue
                    key = f"{self.gh.repo.casefold()}#{n}"
                    row = self.store.parents.get(key)
                    # A crash after the registry save but before schedule save is
                    # repaired from the registry; never reserve a second parent.
                    if (
                        row
                        and managed(row)
                        and row["dispatch"].get("source_run") == run["id"]
                    ):
                        run["candidates"][str(n)] = {
                            "state": "assigned",
                            "reason": "registry予約済み",
                        }
                        continue
                    if row and not released_reservation(row):
                        run["candidates"][str(n)] = {
                            "state": "existing_owner",
                            "reason": "既存担当は自動移管しない",
                        }
                        continue
                    if (
                        sum(occupies(r) for r in self.store.parents.values())
                        >= state["config"]["max_parents"]
                    ):
                        run["candidates"][str(n)] = {
                            "state": "capacity_wait",
                            "reason": "開発親の実行枠待ち",
                        }
                        continue
                    try:
                        # Required order, in this SAME Store.locked interval.
                        judgement.assert_candidate_fresh(
                            self.gh,
                            self.store,
                            Path(state["config"]["cwd"]),
                            snapshot,
                            candidate,
                        )
                        session = ParentSession(self.gh, self.store, n, None)
                        auth = validate_candidate(
                            session, candidate["basis_sha256"], candidate["generation"]
                        )
                    except ValueError as exc:
                        run["candidates"][str(n)] = {
                            "state": "stale",
                            "reason": f"{exc}; 次の予定回または明示実行で再判断する",
                        }
                        continue
                    assignment_id = uuid.uuid4().hex
                    reservation = f"dispatcher-reservation:{assignment_id}"
                    previous = row
                    row = {
                        **(previous or {}),
                        "session": reservation,
                        "status": "active",
                        "generation": candidate["generation"],
                        "tasks": previous["tasks"] if previous else {},
                        "reviews": previous["reviews"] if previous else [],
                        "launch_policy": {
                            "schema_version": 1,
                            "max_parallel_children": state["config"]["max_children"],
                            "reason": "自律ループの実行枠に合わせた予約上限。実runtime上限はdispatcherが別途検査する。",
                        },
                        "dispatch": {
                            "schema_version": 1,
                            "assignment_id": assignment_id,
                            "source_run": run["id"],
                            "phase": "reserved",
                            "session": reservation,
                            "candidate": candidate,
                            "authorization": auth,
                            "child_limit": state["config"]["max_children"],
                            "jobs": {},
                            "active_job": None,
                            "events": [],
                            "reason": "起動予約済み",
                        },
                    }
                    if previous:
                        row["dispatch_history"] = [
                            *previous.get("dispatch_history", []),
                            previous["dispatch"],
                        ]
                    self._job(key, row)
                    self.store.parents[key] = row
                    self.store.save()
                    run["candidates"][str(n)] = {
                        "state": "assigned",
                        "reason": "registryへ予約保存済み",
                    }
                    assigned.append(n)
            self.schedule.write(state)
        return assigned

    def accept(self, inputs: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
        identity = session_identity(env)
        if (
            set(inputs)
            != {
                "schema_version",
                "repo",
                "parent",
                "assignment_id",
                "job_id",
                "authorization",
                "candidate",
                "parent_state_dir",
                "dispatcher_state_dir",
                "runtime_dir",
                "max_implementation_children",
            }
            or type(inputs["schema_version"]) is not int
            or inputs["schema_version"] != 1
            or inputs["repo"] != self.gh.repo.casefold()
        ):
            raise ValueError("invalid bootstrap input schema")
        with self.store.locked():
            key = f"{self.gh.repo.casefold()}#{number(inputs['parent'])}"
            row = self.store.parents[key]
            dispatch = managed(row)
            if (
                not dispatch
                or inputs["schema_version"] != 1
                or inputs["repo"] != self.gh.repo.casefold()
            ):
                raise ValueError("invalid bootstrap assignment")
            job = dispatch["jobs"][dispatch["active_job"]]
            if (
                dispatch["assignment_id"] != inputs["assignment_id"]
                or dispatch["authorization"] != inputs["authorization"]
                or dispatch["candidate"] != inputs["candidate"]
                or inputs["parent_state_dir"] != str(self.store.directory.resolve())
                or inputs["dispatcher_state_dir"] != str(self.schedule.directory)
                or inputs["runtime_dir"] != str(Path(job["directory"]) / "runtime")
                or type(inputs["max_implementation_children"]) is not int
                or inputs["max_implementation_children"] != dispatch["child_limit"]
                or job["id"] != inputs["job_id"]
                or job["phase"] != "starting"
                or env.get("AGENT_RUNTIME_RUN_DIR")
                != str(Path(job["directory"]) / "runtime")
                or env.get("AGENT_RUNTIME_TASK_ID") != job["task_id"]
            ):
                raise ValueError("bootstrap runtime/assignment mismatch")
            evidence = Runtime(Path(job["runner"])).inspect(
                Path(job["directory"]) / "runtime"
            )
            self._check_contract(job, evidence["record"])
            actual = evidence["record"]["runtime"].get("session_id")
            if identity != f"claude:{actual}" or not evidence["summary"]["agent_alive"]:
                raise ValueError(
                    "bootstrap identity must match the live Claude runtime"
                )
            if (
                not row["session"].startswith("dispatcher-reservation:")
                and row["session"] != identity
            ):
                raise ValueError("never replace an existing runtime session")
            self._authorized(key, row)
            if any(
                k != key
                and r["session"] == identity
                and r["status"] in {"active", "handoff_pending"}
                for k, r in self.store.parents.items()
            ):
                raise ValueError("runtime already owns another parent")
            row["session"] = dispatch["session"] = identity
            dispatch.update(phase="running", reason="runtime identity確認済み")
            event(dispatch, "parent_accepted", job_id=job["id"])
            self.store.save()
            return {
                "parent": inputs["parent"],
                "accepted": True,
                "authorization": dispatch["authorization"],
            }

    @staticmethod
    def _check_contract(job: dict[str, Any], record: dict[str, Any]) -> None:
        contract = record.get("contract", {})
        settings = contract.get("settings", {})
        if (
            contract.get("cwd") != job["cwd"]
            or contract.get("task_id") != job["task_id"]
            or contract.get("mode") != "write"
            or settings.get("role") != job["role"]
            or settings.get("profile") is not None
            or settings.get("agent")
            != ("claude" if job["role"] == "parent" else "codex")
        ):
            raise ValueError("runner contract does not match the reserved job")

    def start_child(
        self,
        parent: int,
        child: int,
        delegation_id: str,
        prompt: str,
        env: dict[str, str],
        model: str | None = None,
    ) -> str:
        with self.store.locked():
            key = f"{self.gh.repo.casefold()}#{parent}"
            row = self.store.parents[key]
            dispatch = managed(row)
            if (
                not dispatch
                or row["session"] != session_identity(env)
                or dispatch["phase"] != "running"
            ):
                raise ValueError(
                    "only the accepted running parent may launch implementation workers"
                )
            self._authorized(key, row)
            task = row["tasks"].get(str(child), {})
            if (
                not task.get("delegated")
                or task.get("delegation_id") != delegation_id
                or task.get("worker_ended")
                or task.get("stopped")
                or task.get("authorization") != dispatch["authorization"]
                or task.get("agent") != "codex"
            ):
                raise ValueError("current Codex delegation reservation is required")
            related = [j for j in dispatch["jobs"].values() if j["child"] == child]
            if any(j["phase"] != "ended" for j in related):
                raise ValueError(
                    "previous child runtime is still reserved/live/unreconciled"
                )
            if any(j["delegation_id"] == delegation_id for j in related):
                raise ValueError(
                    "delegation already ran; explicitly re-delegate after checking its result"
                )
            runtime_count = sum(
                j["role"] == "implementation" and j["phase"] != "ended"
                for j in dispatch["jobs"].values()
            )
            if runtime_count >= dispatch["child_limit"]:
                raise ValueError("implementation runtime limit reached")
            # Unlaunched reservations are counted by launch_plan, while this
            # check independently enforces the actual worker limit.
            if len(children_pending(row)) > dispatch["child_limit"]:
                raise ValueError("implementation reservation limit exceeded")
            resume_from = None
            if related:
                previous = max(related, key=lambda j: j["sequence"])
                resume_from = self.resume_source(row, previous)
            job = self._job(
                key,
                row,
                child=child,
                delegation_id=delegation_id,
                prompt=prompt,
                model=model,
                resume_from=resume_from,
            )
            self.store.save()
            return str(job["id"])

    def resume(self, parent: int, reason: str) -> None:
        with self.store.locked():
            key = f"{self.gh.repo.casefold()}#{parent}"
            row = self.store.parents[key]
            dispatch = managed(row)
            if not dispatch or dispatch["phase"] not in {
                "waiting",
                "failed",
                "stopped",
            }:
                raise ValueError("resume requires an idle managed parent")
            self._authorized(key, row)
            previous = dispatch["jobs"][dispatch["active_job"]]
            if self.resume_source(row, previous) is None or row["session"].startswith(
                "dispatcher-reservation:"
            ):
                raise ValueError(
                    "no accepted session to resume; never create a replacement implicitly"
                )
            dispatch.update(
                phase="resume_pending", reason=nonempty(reason, "resume reason")
            )
            event(dispatch, "resume_requested", reason=reason)
            self.store.save()

    def _finish(
        self,
        key: str,
        row: dict[str, Any],
        job: dict[str, Any],
        evidence: dict[str, Any],
    ) -> None:
        dispatch = row["dispatch"]
        record = evidence["record"]
        self._check_contract(job, record)
        job["phase"] = "ended"
        event(
            dispatch,
            "runtime_ended",
            job_id=job["id"],
            child=job["child"],
            reason=record["turn_end_reason"],
        )
        if job["role"] == "implementation":
            task = row["tasks"].get(str(job["child"]), {})
            if task.get("delegation_id") == job["delegation_id"]:
                task["worker_ended"] = {
                    "delegation_id": job["delegation_id"],
                    "evidence": f"dispatcher runtime {record['run_id']}: terminal result, cleanup_complete, no live runner/CLI/group",
                }
            dispatch["wake_requested"] = True
            if dispatch["phase"] == "waiting":
                dispatch.update(
                    phase="resume_pending", reason="子runtimeの終了イベントを確認する"
                )
            return
        if dispatch["phase"] == "stopped":
            return
        if record["result"] != "completed" or row["session"].startswith(
            "dispatcher-reservation:"
        ):
            dispatch.update(
                phase="failed",
                reason="親runtime失敗または担当未受理。記録を確認して明示再開する",
            )
            return
        try:
            response = load(Path(job["directory"]) / "runtime/final.md")
            if (
                set(response)
                != {"schema_version", "state", "reason", "resume_condition"}
                or type(response["schema_version"]) is not int
                or response["schema_version"] != 1
                or response["state"] not in {"waiting", "completed", "failed"}
            ):
                raise ValueError("invalid parent event schema")
            nonempty(response["reason"], "parent event reason")
            nonempty(response["resume_condition"], "parent resume condition")
            dispatch.update(
                phase=response["state"],
                reason=response["reason"],
                resume_condition=response["resume_condition"],
            )
            if response["state"] == "waiting" and dispatch.get("wake_requested"):
                dispatch.update(
                    phase="resume_pending", reason="子runtimeの終了イベントを確認する"
                )
            if response["state"] == "completed":
                if children_pending(row):
                    raise ValueError(
                        "cannot finish a parent while child reservations/runtimes remain"
                    )
                row["status"] = "ended"
            event(
                dispatch,
                "parent_" + response["state"],
                job_id=job["id"],
                reason=response["reason"],
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            dispatch.update(phase="failed", reason=str(exc))

    def reconcile(self, *, stop: bool = False, parent: int | None = None) -> None:
        with self.store.locked():
            state = self.schedule.read()
            for key, row in self.store.parents.items():
                dispatch = managed(row)
                if not dispatch or (
                    parent is not None and key != f"{self.gh.repo.casefold()}#{parent}"
                ):
                    continue
                if stop:
                    dispatch.update(phase="stopped", reason="明示停止。担当は保持する")
                    event(dispatch, "stop_requested")
                for job in dispatch["jobs"].values():
                    if job["phase"] == "ended":
                        continue
                    directory = Path(job["directory"])
                    if job["phase"] == "reserved":
                        if stop:
                            self.cancel_reservation(
                                row, job, "cancelled before runtime launch"
                            )
                        continue
                    # A live worker (or inherited launcher FD) may still be in
                    # Popen/save. Absence of a runtime file is never end evidence.
                    with gate(directory / "lifetime.lock") as fd:
                        busy = fd is None
                    path = directory / "runtime/result.json"
                    if not path.exists():
                        if stop and alive(job["worker"]):
                            os.kill(job["worker"]["pid"], signal.SIGTERM)
                        if not busy and not alive(job["worker"]):
                            job.update(
                                phase="recovery_required",
                                diagnostic="起動結果欠落。別sessionを起動せず実プロセスを照合する",
                            )
                            dispatch.update(
                                phase="recovery_required", reason=job["diagnostic"]
                            )
                        continue
                    runtime = Runtime(Path(job["runner"]))
                    evidence = runtime.inspect(directory / "runtime", stop=stop)
                    if (
                        runtime.ended(evidence)
                        and not busy
                        and not alive(job["worker"])
                    ):
                        self._finish(key, row, job, evidence)
                if (
                    not stop
                    and state["enabled"]
                    and dispatch["phase"] == "resume_pending"
                ):
                    # A parent's live children retain its slot. An idle parent
                    # without children must reacquire capacity before resuming.
                    used = sum(
                        occupies(r) for k, r in self.store.parents.items() if k != key
                    )
                    if used < state["config"]["max_parents"]:
                        try:
                            self._authorized(key, row)
                            previous = dispatch["jobs"][dispatch["active_job"]]
                            resume_from = self.resume_source(row, previous)
                            if resume_from is None:
                                raise ValueError(
                                    "previous runtime has no resumable session"
                                )
                            self._job(
                                key,
                                row,
                                resume_from=resume_from,
                            )
                        except (ValueError, OSError) as exc:
                            dispatch.update(phase="failed", reason=str(exc))
            self.store.save()

    def launch_reserved(self) -> None:
        with self.store.locked():
            if not self.schedule.read()["enabled"]:
                return
            for key, row in self.store.parents.items():
                dispatch = managed(row)
                if not dispatch or dispatch["phase"] in {
                    "stopped",
                    "recovery_required",
                }:
                    continue
                for job in dispatch["jobs"].values():
                    if job["phase"] == "reserved":
                        Runtime(Path(job["runner"])).spawn(
                            self.store.directory.resolve(),
                            key,
                            job["id"],
                            Path(job["directory"]),
                            schedule_dir=self.schedule.directory,
                        )

    def status(self, now: datetime) -> dict[str, Any]:
        with self.store.locked():
            state = self.schedule.read()
            parents = []
            for key, row in self.store.parents.items():
                dispatch = managed(row)
                if dispatch:
                    parents.append(
                        {
                            "parent": key,
                            "assignment": row["status"],
                            "phase": dispatch["phase"],
                            "occupies_slot": occupies(row),
                            "children_pending": children_pending(row),
                            "max_implementation_children": dispatch["child_limit"],
                            "reason": dispatch["reason"],
                            "jobs": [
                                {
                                    "id": j["id"],
                                    "role": j["role"],
                                    "child": j["child"],
                                    "phase": j["phase"],
                                }
                                for j in dispatch["jobs"].values()
                            ],
                        }
                    )
            return {
                "schema_version": 1,
                "enabled": state["enabled"],
                "max_development_parents": state["config"]["max_parents"],
                "default_max_implementation_children": state["config"]["max_children"],
                "used_parent_slots": sum(
                    occupies(r) for r in self.store.parents.values()
                ),
                "last_slot": state["cursor"],
                "next_slot": next_slot(now).isoformat(),
                "due_slot": latest_slot(now).isoformat()
                if latest_slot(now).isoformat() > state["cursor"]
                else None,
                "active_judgement": state["active"],
                "judgements": [
                    {
                        **{
                            k: r[k]
                            for k in (
                                "id",
                                "slot",
                                "explicit",
                                "phase",
                                "diagnostic",
                                "candidates",
                            )
                        },
                        "deferred": read_result(Path(r["directory"]))["deferred"]
                        if (Path(r["directory"]) / "result.json").exists()
                        else None,
                    }
                    for r in state["runs"]
                ],
                "parents": parents,
            }
