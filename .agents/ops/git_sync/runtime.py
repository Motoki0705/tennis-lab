"""Quiesce resident services without overriding the operator's OFF state."""

from __future__ import annotations

import json
import re
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ops_control.control import Controller
from ops_control.systemd import ControlError
from ops_control.targets import TARGETS

from .git import Skip

SUPERVISOR = "tennis-lab-dispatch-supervisor.service"
BOARD = "tennis-lab-board-web.service"


def managed_busy(state: dict[str, Any]) -> bool:
    busy = state["active_judgement"] is not None
    for parent in state["parents"]:
        if (
            not isinstance(parent, dict)
            or not isinstance(parent.get("children_pending"), list)
            or not isinstance(parent.get("jobs"), list)
            or type(parent.get("occupies_slot")) is not bool
            or any(
                not isinstance(job, dict)
                or job.get("phase")
                not in {"reserved", "starting", "ended", "recovery_required"}
                for job in parent["jobs"]
            )
        ):
            raise ControlError("dispatcher status: incomplete managed-job evidence")
        busy |= bool(
            parent["children_pending"]
            or parent["occupies_slot"]
            or any(job["phase"] != "ended" for job in parent["jobs"])
        )
    return busy


class Maintenance:
    def __init__(self, root: Path, directory: Path, event: Callable[..., None]) -> None:
        self.control = Controller(root)
        self.ticket = directory / "resume.json"
        self.event = event

    def guarded(self, state: dict[str, Any]) -> bool:
        command = state.get("exec_start") or ""
        root = self.control.repo_root
        return str(root / ".agents/ops/checkout_job.py") in command and bool(
            re.search(r"--repo\s+" + re.escape(str(root)) + r"(?:\s|;|$)", command)
        )

    def status(self, unit: str) -> dict[str, Any] | None:
        # Uninstalled jobs have no running service. Other lookup failures are
        # errors, including a missing user manager, never evidence of OFF.
        result = self.control.systemd.invoke(
            "show", unit, "--property=LoadState,ActiveState"
        )
        properties = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        if (
            properties.get("LoadState") == "not-found"
            and properties.get("ActiveState") == "inactive"
        ):
            self.control.systemd.require_manager()
            return None
        return self.control.systemd.status(unit)

    def finite_idle(self) -> None:
        running = []
        unguarded = []
        units = set()
        for target in TARGETS.values():
            if target.run_unit and target.name != "git-sync":
                units.add(target.run_unit)
            if target.verify_dispatcher:
                units.update(
                    unit
                    for unit in target.units
                    if unit.endswith(".service") and unit != SUPERVISOR
                )
        for unit in sorted(units):
            state = self.status(unit)
            if state and state["active_state"] not in {"inactive", "failed"}:
                running.append(unit)
            elif state and not self.guarded(state):
                unguarded.append(unit)
        if running:
            raise Skip("finite_jobs_running", units=running)
        if unguarded:
            raise Skip("unguarded_job_units", units=unguarded)

    def pause(self) -> None:
        self.control.systemd.require_manager()
        self.finite_idle()
        supervisor = self.status(SUPERVISOR)
        if supervisor:
            state = self.control.dispatcher("status")
            if managed_busy(state):
                raise Skip("managed_jobs_running", dispatcher=state)
            if supervisor["active_state"] == "active":
                # Type=simple may already be active while its Python launcher
                # is still waiting for our code lock. Older supervisors also
                # lack the maintenance handler. HUP must never kill either.
                try:
                    pid = int(supervisor.get("main_pid") or "0")
                    process = dict(
                        line.split(":", 1)
                        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                        if ":" in line
                    )
                    ready = pid > 0 and bool(
                        int(process["SigCgt"].strip(), 16) & (1 << (signal.SIGHUP - 1))
                    )
                except (OSError, ValueError, KeyError):
                    ready = False
                if not ready:
                    raise Skip("supervisor_not_maintenance_ready")
        residents: list[dict[str, Any]] = []
        for unit in (SUPERVISOR, BOARD):
            service = supervisor if unit == SUPERVISOR else self.status(unit)
            if not service:
                continue
            if not self.guarded(service):
                raise Skip("unguarded_job_units", units=[unit])
            if service["active_state"] not in {"active", "inactive", "failed"}:
                raise Skip("resident_transition", unit=unit, state=service)
            if service["active_state"] == "active":
                residents.append(
                    {
                        "unit": unit,
                        "restart": service["unit_file_state"] == "enabled"
                        and (unit != SUPERVISOR or state["enabled"]),
                    }
                )
        # The ticket survives a killed updater. A subsequent run restores only
        # the services we paused, after rechecking their enabled state.
        self.ticket.write_text(json.dumps(residents))
        for resident in residents:
            unit = resident["unit"]
            if unit == SUPERVISOR:
                self.control.systemd.run(
                    "kill", "--kill-whom=main", "--signal=HUP", unit
                )
                deadline = time.monotonic() + 15
                while True:
                    current = self.status(unit)
                    if current and current["active_state"] == "inactive":
                        break
                    if time.monotonic() >= deadline:
                        raise ControlError("dispatcher maintenance pause not confirmed")
                    time.sleep(0.05)
                # Confirm the post-exit managed state as well, before merging.
                state = self.control.dispatcher("status")
                if managed_busy(state):
                    raise ControlError(
                        "dispatcher jobs appeared during maintenance pause"
                    )
            else:
                self.control.systemd.run("stop", unit)
                current = self.status(unit)
                if not current or current["active_state"] != "inactive":
                    raise ControlError(f"{unit}: maintenance stop not confirmed")
            self.event("resident_paused", unit=unit, restart=resident["restart"])

    def resume(self) -> None:
        if not self.ticket.exists():
            return
        residents = json.loads(self.ticket.read_text())
        if not isinstance(residents, list) or any(
            not isinstance(row, dict)
            or row.get("unit") not in {SUPERVISOR, BOARD}
            or type(row.get("restart")) is not bool
            for row in residents
        ):
            raise ControlError("invalid maintenance resume ticket")
        errors = []
        for resident in residents:
            unit = resident["unit"]
            try:
                current = self.status(unit)
                enabled = bool(current and current["unit_file_state"] == "enabled")
                if unit == SUPERVISOR and enabled:
                    enabled = self.control.dispatcher("status")["enabled"]
                if not resident["restart"] or not enabled:
                    self.event("resident_left_off", unit=unit)
                    continue
                if current and current["active_state"] != "active":
                    self.control.systemd.run("start", unit)
                current = self.status(unit)
                if not current or current["active_state"] != "active":
                    raise ControlError(f"{unit}: restart not confirmed")
                self.event("resident_resumed", unit=unit)
            except (ControlError, OSError, ValueError) as exc:
                errors.append(str(exc))
        if errors:
            raise ControlError("; ".join(errors))
        self.ticket.unlink()
