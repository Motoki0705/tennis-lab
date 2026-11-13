"""Execute target contracts; continue stopping independent targets on failure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .systemd import ControlError, ManagerUnavailable, Systemd
from .targets import Step, Target


class Controller:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.systemd = Systemd(repo_root)

    def dispatcher(self, command: str) -> dict[str, Any]:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.repo_root / ".agents/ops")
        result = subprocess.run(
            [sys.executable, "-m", "dispatcher", command],
            cwd=self.repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ControlError(
                f"dispatcher {command}: {result.stderr.strip() or result.stdout.strip()}; "
                "inspect dispatcher state (first setup requires python -m dispatcher init)"
            )
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ControlError(f"dispatcher {command}: expected a JSON object")
        if command != "status" and data.get("result") != command:
            raise ControlError(
                f"dispatcher {command}: unexpected result {data.get('result')!r}"
            )
        if command == "status" and (
            data.get("schema_version") != 1
            or type(data.get("enabled")) is not bool
            or not isinstance(data.get("parents"), list)
            or "active_judgement" not in data
        ):
            raise ControlError(
                "dispatcher status: incomplete or unsupported status evidence"
            )
        return data

    def step(self, step: Step) -> None:
        if step.tool == "systemctl":
            if step.args[0] == "disable":
                self.systemd.disable(*step.args[1:])
            else:
                self.systemd.run(*step.args)
        else:
            self.dispatcher(*step.args)

    def inspect(
        self,
        target: Target,
        report: dict[str, Any],
        *,
        continue_on_unavailable: bool = False,
    ) -> None:
        for unit in target.units:
            try:
                report["units"].append(self.systemd.status(unit))
            except ManagerUnavailable as exc:
                if not continue_on_unavailable:
                    raise
                report["outcome"] = "unavailable"
                report["errors"].append(f"{unit}: {exc}")
            except (ControlError, OSError, ValueError) as exc:
                report["errors"].append(str(exc))
        if target.verify_dispatcher:
            try:
                report["dispatcher"] = self.dispatcher("status")
            except (ControlError, OSError, ValueError) as exc:
                report["errors"].append(str(exc))

    def verify_off(self, report: dict[str, Any]) -> None:
        for unit in report["units"]:
            # Static oneshot services cannot be enabled. Failed services have
            # exited, but their last failure stays visible in status.
            if unit["unit_file_state"] not in {
                "disabled",
                "static",
                "indirect",
                "linked",
                "linked-runtime",
                "masked",
                "masked-runtime",
            }:
                report["errors"].append(
                    f"{unit['unit']}: remains enabled/unconfirmed ({unit['unit_file_state']})"
                )
            if unit["active_state"] not in {"inactive", "failed"}:
                report["errors"].append(
                    f"{unit['unit']}: remains {unit['active_state']}"
                )
        if "dispatcher" not in report:
            return
        state = report["dispatcher"]
        if state["enabled"]:
            report["errors"].append("dispatcher remains enabled")
        if state["active_judgement"] is not None:
            report["errors"].append("dispatcher judgement remains active/unconfirmed")
        for parent in state["parents"]:
            if (
                not isinstance(parent, dict)
                or not isinstance(parent.get("children_pending"), list)
                or not isinstance(parent.get("jobs"), list)
                or type(parent.get("occupies_slot")) is not bool
                or any(
                    not isinstance(job, dict) or "phase" not in job
                    for job in parent["jobs"]
                )
            ):
                raise ControlError("dispatcher status: incomplete managed-job evidence")
            pending = [job for job in parent["jobs"] if job["phase"] != "ended"]
            if parent["children_pending"] or pending or parent["occupies_slot"]:
                report["errors"].append(
                    f"dispatcher managed parent/children remain unconfirmed: {parent}"
                )

    def operate(self, action: str, target: Target) -> dict[str, Any]:
        report: dict[str, Any] = {
            "target": target.name,
            "kind": target.kind,
            "units": [],
            "errors": [],
            "outcome": "ok",
        }
        if action == "off":
            # A failed shutdown must not skip disabling the supervisor, or
            # leave other targets running in an agents group operation.
            for step in target.off_steps:
                try:
                    self.step(step)
                except (ControlError, OSError, ValueError) as exc:
                    if isinstance(exc, ManagerUnavailable):
                        report["outcome"] = "unavailable"
                    report["errors"].append(str(exc))
        else:
            try:
                if action == "on":
                    for step in target.before_on:
                        self.step(step)
                    self.systemd.run("enable", "--now", *target.enable_units)
                elif action == "run":
                    assert target.run_unit is not None
                    self.systemd.run("start", target.run_unit)
            except ManagerUnavailable:
                raise
            except (ControlError, OSError, ValueError) as exc:
                report["errors"].append(str(exc))
        self.inspect(target, report, continue_on_unavailable=action == "off")
        try:
            if action == "off":
                self.verify_off(report)
            elif action == "on":
                for unit in report["units"]:
                    if unit["unit"] in target.enable_units and (
                        unit["unit_file_state"] != "enabled"
                        or unit["active_state"] != "active"
                    ):
                        report["errors"].append(
                            f"{unit['unit']}: persistent enable/start not confirmed"
                        )
                if (
                    target.verify_dispatcher
                    and "dispatcher" in report
                    and not report["dispatcher"]["enabled"]
                ):
                    report["errors"].append("dispatcher enable not confirmed")
        except ControlError as exc:
            report["errors"].append(str(exc))
        if report["outcome"] != "unavailable":
            report["outcome"] = "partial_failure" if report["errors"] else "ok"
        return report

    def install(self, targets: tuple[Target, ...], *, dry_run: bool) -> int:
        command = ["bash", str(self.repo_root / ".agents/ops/install_units.sh")]
        if dry_run:
            command.append("--dry-run")
        for unit in dict.fromkeys(unit for target in targets for unit in target.units):
            command.extend(("--unit", unit))
        return subprocess.run(command, cwd=self.repo_root, check=False).returncode

    def logs(self, targets: tuple[Target, ...], *, lines: int, follow: bool) -> int:
        command = ["journalctl", "--user", "--lines", str(lines)]
        command.append("--follow" if follow else "--no-pager")
        for unit in dict.fromkeys(unit for target in targets for unit in target.units):
            command.extend(("--unit", unit))
        return subprocess.run(command, cwd=self.repo_root, check=False).returncode
