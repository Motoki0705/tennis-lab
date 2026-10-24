"""Idempotent dispatch controller and independently callable deadline watchdog."""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Callable
from typing import Any

from lib.opsproc import CommandError

from .api import CycleAPI
from .github import Actions
from .hooks import CommandHooks
from .model import LIVE, generation, instant, positive, sha, window

Hook = Callable[[str, str, dict[str, Any] | None, float], dict[str, Any]]
IO_ERRORS = (
    ValueError,
    OSError,
    KeyError,
    TypeError,
    subprocess.SubprocessError,
    CommandError,
)


class Controller:
    def __init__(
        self, api: CycleAPI, actions: Actions, *, hooks: Hook | None = None
    ) -> None:
        self.api = api
        self.store = api.store
        self.actions = actions
        self.hooks = hooks if hooks is not None else CommandHooks(api)

    def enable(self) -> None:
        for role in ("observer", "workers"):
            result = self.hooks(role, "probe", None, 20)
            if result != {"schema_version": 1, "ready": True, "role": role}:
                raise ValueError(f"daily_ci {role} integration is not ready")
        with self.store.repository_lock(), self.store.transaction() as data:
            data["enabled"] = True

    def disable(self) -> None:
        with self.store.transaction() as data:
            data["enabled"] = False
            for cycle in data["cycles"].values():
                if cycle["status"] in LIVE:
                    self.api.end(cycle, "disabled", "operator_disabled")

    def budget(self, cycle_id: str) -> float:
        permit = self.api.permit_worker(cycle_id)
        remaining = (instant(permit["deadline_at"]) - self.api.clock()).total_seconds()
        if remaining <= 0:
            raise ValueError("daily_ci deadline reached")
        return min(20.0, remaining)

    def new_cycle(self, data: dict[str, Any], status: str) -> dict[str, Any]:
        now = self.api.clock()
        cycle_id, opens, deadline = window(now)
        previous = list(data["cycles"].values())
        cycle: dict[str, Any] = {
            "cycle_id": cycle_id,
            "date": opens.date().isoformat(),
            "opens_at": opens.isoformat(),
            "deadline_at": deadline.isoformat(),
            "created_at": now.isoformat(),
            "status": status,
            "cutoff_sha": None,
            "target_sha": None,
            "observed_main_sha": None,
            "baseline_sha": (
                data["last_success"]["target_sha"]
                if data["last_success"]
                else data["baseline_sha"]
            ),
            "workflows": data["workflows"],
            "generations": [],
            "repair_prs": {},
            "workers": {},
            "unresolved": copy.deepcopy(previous[-1]["unresolved"]) if previous else [],
            "carryover": {
                "cycle_id": previous[-1]["cycle_id"],
                "repair_prs": copy.deepcopy(previous[-1]["repair_prs"]),
                "target_sha": previous[-1]["target_sha"],
                "status": previous[-1]["status"],
            }
            if previous
            else None,
            "errors": [],
            "end_reason": "host_missed_window" if status == "missed" else None,
            "freeze_released_at": now.isoformat() if status == "missed" else None,
            "stop_complete": status == "missed",
        }
        data["cycles"][cycle_id] = cycle
        return cycle

    def start(self) -> str | None:
        now = self.api.clock()
        cycle_id, opens, deadline = window(now)
        with self.store.transaction() as data:
            if not data["enabled"]:
                return None
            if now < opens:
                return None
            if now >= deadline:
                if cycle_id not in data["cycles"]:
                    self.new_cycle(data, "missed")
                return None
            cycle = data["cycles"].get(cycle_id)
            if cycle is None:
                cycle = self.new_cycle(data, "freezing")
            if cycle["status"] not in {"freezing", "active"}:
                return None
        # repository_lock is held by tick. Freeze is durable before the main GET.
        main_sha = self.actions.main_sha()
        with self.store.transaction() as data:
            cycle = data["cycles"][cycle_id]
            if self.api.clock() >= deadline or cycle["status"] not in {
                "freezing",
                "active",
            }:
                return None
            cycle["observed_main_sha"] = main_sha
            if cycle["status"] == "freezing":
                cycle.update(status="active", cutoff_sha=main_sha, target_sha=main_sha)
                cycle["generations"].append(
                    generation(self.store.repository, cycle, main_sha, "initial")
                )
            elif main_sha != cycle["target_sha"]:
                cycle.update(status="blocked", end_reason="external_main_changed")
                cycle["errors"].append("external_main_changed")
                for worker in cycle["workers"].values():
                    if worker["state"] != "stopped":
                        worker["state"] = "stop_requested"
                return None
        return cycle_id

    def cycle(self, cycle_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self.api.snapshot()["cycles"][cycle_id]
        return result

    def reconcile(self, cycle_id: str, number: int, workflow: str) -> None:
        cycle = self.cycle(cycle_id)
        revision = cycle["generations"][number - 1]
        request = revision["requests"][workflow]
        if request["state"] == "planned":
            return
        # Even a returned run ID is checked against the marker and all run identities.
        candidates = self.actions.runs(workflow, request["requested_at"])
        matches = [
            r
            for r in candidates
            if r.get("display_title") == f"daily-ci:{request['request_id']}"
        ]
        if len(matches) > 1:
            with self.store.transaction() as data:
                current = data["cycles"][cycle_id]["generations"][number - 1][
                    "requests"
                ][workflow]
                current["conflicting_run_ids"] = [positive(r["id"]) for r in matches]
            raise ValueError("multiple runs for one dispatch request")
        if not matches:
            if request["run_id"] is None:
                return  # Absence is not proof that POST was never accepted. Never resend.
            matches = [self.actions.run(request["run_id"])]
        row = matches[0]
        identifier, attempt = positive(row["id"]), positive(row["run_attempt"])
        with self.store.transaction() as data:
            current = data["cycles"][cycle_id]["generations"][number - 1]["requests"][
                workflow
            ]
            current["seen_run_ids"] = sorted(
                set(current.get("seen_run_ids", [])) | {identifier}
            )
        if (
            row.get("display_title") != f"daily-ci:{request['request_id']}"
            or row.get("event") != "workflow_dispatch"
            or row.get("head_branch") != "main"
            or row.get("path", "").split("@")[0] != f".github/workflows/{workflow}"
            or sha(row.get("head_sha")) != revision["target_sha"]
            or (request["run_id"] is not None and request["run_id"] != identifier)
            or (request["attempt"] is not None and attempt < request["attempt"])
            or row.get("status")
            not in {
                "queued",
                "in_progress",
                "completed",
                "waiting",
                "pending",
                "requested",
            }
        ):
            raise ValueError("dispatch run identity/SHA/attempt mismatch")
        with self.store.transaction() as data:
            current = data["cycles"][cycle_id]["generations"][number - 1]["requests"][
                workflow
            ]
            if current["attempt"] != attempt:
                current["observation"] = None
            current.update(
                state="bound",
                run_id=identifier,
                attempt=attempt,
                remote_status=row["status"],
                remote_conclusion=row.get("conclusion"),
                error=None,
            )

    def dispatch(self, cycle_id: str, workflow: str) -> None:
        cycle = self.cycle(cycle_id)
        number = len(cycle["generations"])
        revision = cycle["generations"][-1]
        request = revision["requests"][workflow]
        if request["state"] != "planned":
            self.reconcile(cycle_id, number, workflow)
            return
        self.budget(cycle_id)
        if self.actions.main_sha() != revision["target_sha"]:
            raise ValueError("main changed before workflow dispatch")
        with self.store.transaction() as data:
            current_cycle = self.api.active(data, cycle_id)
            current = current_cycle["generations"][number - 1]["requests"][workflow]
            current.update(
                state="dispatch_unknown", requested_at=self.api.clock().isoformat()
            )
        inputs = {
            "cycle_id": cycle_id,
            "generation": str(number),
            "request_id": request["request_id"],
            "target_sha": revision["target_sha"],
            "baseline_sha": cycle["baseline_sha"],
            "deadline_at": cycle["deadline_at"],
        }
        # Durable uncertainty precedes the single POST, including crashes just before it.
        identifier = self.actions.dispatch(workflow, inputs, self.budget(cycle_id))
        with self.store.transaction() as data:
            current = data["cycles"][cycle_id]["generations"][number - 1]["requests"][
                workflow
            ]
            current.update(state="submitted", run_id=identifier)
        self.reconcile(cycle_id, number, workflow)

    def finish_success(self, cycle_id: str) -> None:
        cycle = self.cycle(cycle_id)
        for workflow in cycle["workflows"]:
            self.reconcile(cycle_id, len(cycle["generations"]), workflow)
        main_sha = self.actions.main_sha()
        with self.store.transaction() as data:
            cycle = self.api.active(data, cycle_id)
            cycle["observed_main_sha"] = main_sha
            if main_sha != cycle["target_sha"]:
                cycle.update(status="blocked", end_reason="external_main_changed")
                cycle["errors"].append("external_main_changed")
                return
            merge_unknown = any(
                repair["merge_state"] in {"attempting", "unknown"}
                for repair in cycle["repair_prs"].values()
            )
            if (
                cycle["unresolved"]
                or merge_unknown
                or not all(
                    r["state"] == "bound"
                    and r["remote_status"] == "completed"
                    and r["remote_conclusion"] == "success"
                    and r["observation"] is not None
                    and r["observation"]["attempt"] == r["attempt"]
                    and r["observation"]["result"] == "success"
                    for r in cycle["generations"][-1]["requests"].values()
                )
            ):
                return
            self.api.end(cycle, "success", "all_required_checks_verified")
            data["last_success"] = {
                "cycle_id": cycle_id,
                "target_sha": cycle["target_sha"],
                "verified_at": self.api.clock().isoformat(),
            }

    def tick(self) -> dict[str, Any]:
        errors = self.watchdog()
        with self.store.lock("controller", blocking=False):
            with self.store.repository_lock():
                cycle_id = self.start()
                if cycle_id is None:
                    return {"errors": errors, "state": self.api.snapshot()}
                for workflow in self.cycle(cycle_id)["workflows"]:
                    try:
                        self.dispatch(cycle_id, workflow)
                    except IO_ERRORS as exc:
                        errors.append(self.error(cycle_id, f"dispatch:{workflow}", exc))
            try:
                response = self.hooks(
                    "observer", "collect", self.cycle(cycle_id), self.budget(cycle_id)
                )
                if (
                    set(response) != {"schema_version", "results", "unresolved"}
                    or response["schema_version"] != 1
                ):
                    raise ValueError("invalid observer response")
                for result in response["results"]:
                    self.api.record_result(result)
                self.api.unresolved(cycle_id, response["unresolved"])
                response = self.hooks(
                    "workers", "reconcile", self.cycle(cycle_id), self.budget(cycle_id)
                )
                if response != {"schema_version": 1, "acknowledged": True}:
                    raise ValueError("worker reconciliation was not acknowledged")
            except IO_ERRORS as exc:
                errors.append(self.error(cycle_id, "integration", exc))
            if not errors:
                try:
                    with self.store.repository_lock():
                        self.finish_success(cycle_id)
                except IO_ERRORS as exc:
                    errors.append(self.error(cycle_id, "verification", exc))
        errors.extend(self.watchdog())
        return {"errors": errors, "state": self.api.snapshot()}

    def error(self, cycle_id: str, operation: str, exc: BaseException) -> str:
        message = f"{operation}: {type(exc).__name__}: {str(exc)[:1000]}"
        with self.store.transaction() as data:
            cycle = data["cycles"][cycle_id]
            if not cycle["errors"] or cycle["errors"][-1] != message:
                cycle["errors"].append(message)
        return message

    def watchdog(self) -> list[str]:
        self.api.expire()
        errors = []
        # No controller/repository mutex: a hung controller or merge cannot hold expiry.
        with self.store.lock("watchdog", blocking=False):
            for cycle in self.api.snapshot()["cycles"].values():
                if (
                    cycle["status"] not in {"success", "expired", "disabled", "blocked"}
                    or cycle["stop_complete"]
                ):
                    continue
                cycle_id = cycle["cycle_id"]
                workers_stopped = False
                try:
                    response = self.hooks("workers", "stop", cycle, 20)
                    if (
                        set(response) != {"schema_version", "acknowledged", "workers"}
                        or response["schema_version"] != 1
                        or response["acknowledged"] is not True
                        or set(response["workers"]) != set(cycle["workers"])
                    ):
                        raise ValueError("worker stop acknowledgement is incomplete")
                    with self.store.transaction() as data:
                        for worker_id, outcome in response["workers"].items():
                            if outcome.get("state") not in {
                                "stopped",
                                "stop_unknown",
                            } or not outcome.get("evidence"):
                                raise ValueError(
                                    "worker stop requires termination evidence"
                                )
                            data["cycles"][cycle_id]["workers"][worker_id].update(
                                outcome
                            )
                    workers_stopped = all(
                        w["state"] == "stopped" for w in response["workers"].values()
                    )
                except IO_ERRORS as exc:
                    errors.append(self.error(cycle_id, "stop:workers", exc))
                    with self.store.transaction() as data:
                        for worker in data["cycles"][cycle_id]["workers"].values():
                            if worker["state"] != "stopped":
                                worker["state"] = "stop_unknown"
                runs_stopped = True
                for revision in cycle["generations"]:
                    for workflow in revision["requests"]:
                        try:
                            stopped = self.stop_run(
                                cycle_id, revision["number"], workflow
                            )
                            runs_stopped &= stopped
                        except IO_ERRORS as exc:
                            runs_stopped = False
                            errors.append(self.error(cycle_id, f"stop:{workflow}", exc))
                            with self.store.transaction() as data:
                                request = data["cycles"][cycle_id]["generations"][
                                    revision["number"] - 1
                                ]["requests"][workflow]
                                request["stop_state"] = "stop_unknown"
                with self.store.transaction() as data:
                    data["cycles"][cycle_id]["stop_complete"] = (
                        workers_stopped and runs_stopped
                    )
        return errors

    def stop_run(self, cycle_id: str, number: int, workflow: str) -> bool:
        request = self.cycle(cycle_id)["generations"][number - 1]["requests"][workflow]
        identity_confirmed = True
        if request["state"] != "planned":
            try:
                self.reconcile(cycle_id, number, workflow)
            except IO_ERRORS as exc:
                self.error(cycle_id, f"stop:reconcile:{workflow}", exc)
                identity_confirmed = False
            request = self.cycle(cycle_id)["generations"][number - 1]["requests"][
                workflow
            ]
        identifiers = set(request.get("conflicting_run_ids", [])) | set(
            request.get("seen_run_ids", [])
        )
        if request["run_id"] is not None:
            identifiers.add(request["run_id"])
        stopped = identity_confirmed and (
            request["state"] == "planned" or bool(identifiers)
        )
        for identifier in identifiers:
            row = self.actions.run(identifier)
            if row["status"] != "completed":
                try:
                    self.actions.cancel(identifier)
                finally:
                    # Acceptance, lost response and HTTP 409 are not termination proof.
                    row = self.actions.run(identifier)
                stopped &= row["status"] == "completed"
        with self.store.transaction() as data:
            current = data["cycles"][cycle_id]["generations"][number - 1]["requests"][
                workflow
            ]
            current["stop_state"] = "stopped" if stopped else "stop_unknown"
        return stopped
