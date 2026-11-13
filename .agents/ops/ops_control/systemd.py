"""Read systemd properties without conflating a missing user bus with OFF."""

from __future__ import annotations

import subprocess
from pathlib import Path

MANAGER_HINT = (
    "Check the user login, XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS. "
    "For a user manager that runs without login, run "
    "`sudo loginctl enable-linger <user>` and retry."
)
PROPERTIES = {
    "fragment_path": "FragmentPath",
    "load_state": "LoadState",
    "unit_file_state": "UnitFileState",
    "active_state": "ActiveState",
    "sub_state": "SubState",
    "last_result": "Result",
    "exec_main_status": "ExecMainStatus",
    "next_elapse_realtime": "NextElapseUSecRealtime",
    "next_elapse_monotonic": "NextElapseUSecMonotonic",
    "last_trigger": "LastTriggerUSec",
}


class ControlError(RuntimeError):
    pass


class ManagerUnavailable(ControlError):
    pass


class Systemd:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def invoke(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["systemctl", "--user", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def require_manager(self) -> None:
        try:
            result = self.invoke("show", "--property=Version", "--value")
        except OSError as exc:
            raise ManagerUnavailable(f"{exc}. {MANAGER_HINT}") from exc
        if result.returncode or not result.stdout.strip():
            detail = result.stderr.strip() or "no user manager Version returned"
            raise ManagerUnavailable(f"{detail}. {MANAGER_HINT}")

    def run(self, *args: str) -> None:
        result = self.invoke(*args)
        if result.returncode:
            # A manager may disappear after the initial probe. Distinguish this
            # from an ordinary failing unit operation without parsing locale text.
            self.require_manager()
            raise ControlError(
                f"systemctl --user {' '.join(args)}: "
                f"{result.stderr.strip() or result.stdout.strip() or result.returncode}"
            )

    def disable(self, *args: str) -> None:
        # disable removes ALL links to the unit, including the direct link
        # installed from this repo. Preserve the resolved source before removal
        # and relink it without any wants/enable links, so on/status still work.
        sources = []
        errors: list[ControlError | OSError] = []
        for unit in args:
            if unit.endswith((".service", ".timer")):
                try:
                    fragment = self.status(unit)["fragment_path"]
                    if not fragment or not Path(fragment).is_absolute():
                        raise ControlError(f"{unit}: missing absolute FragmentPath")
                    sources.append(str(Path(fragment).resolve(strict=True)))
                except (ControlError, OSError) as exc:
                    errors.append(exc)
        # Even failed source discovery must not skip the stop request. Keep
        # manager loss distinct if the bus reconnects before later operations.
        try:
            self.run("disable", *args)
        except (ControlError, OSError) as exc:
            errors.append(exc)
        for source in sources:
            try:
                self.run("link", source)
            except (ControlError, OSError) as exc:
                errors.append(exc)
        if errors:
            error_type = (
                ManagerUnavailable
                if any(isinstance(error, ManagerUnavailable) for error in errors)
                else ControlError
            )
            raise error_type("; ".join(str(error) for error in errors))

    def status(self, unit: str) -> dict[str, str | None]:
        result = self.invoke(
            "show",
            unit,
            "--all",
            "--no-pager",
            "--property=" + ",".join(PROPERTIES.values()),
        )
        properties = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        if result.returncode or properties.get("LoadState") != "loaded":
            self.require_manager()
            raise ControlError(
                f"{unit}: load_state={properties.get('LoadState', 'unknown')}; "
                f"{result.stderr.strip() or 'unit is not loaded; run spin ops install'}"
            )
        required = {"LoadState", "UnitFileState", "ActiveState", "SubState"}
        required.update(
            {"NextElapseUSecRealtime", "NextElapseUSecMonotonic"}
            if unit.endswith(".timer")
            else {"Result", "ExecMainStatus"}
        )
        missing = required - properties.keys()
        if missing:
            raise ControlError(
                f"{unit}: missing systemd properties: {', '.join(sorted(missing))}"
            )
        return {
            "unit": unit,
            **{
                field: properties.get(prop) or None
                for field, prop in PROPERTIES.items()
            },
        }
