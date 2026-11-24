"""Small, explicitly targeted Herdr CLI adapter."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeGuard


class HerdrError(ValueError):
    """The Herdr CLI, connection, or response could not be verified."""


def _valid_pane(row: object) -> TypeGuard[dict[str, Any]]:
    if not isinstance(row, dict):
        return False
    tokens = row.get("tokens", {})
    return (
        all(
            isinstance(row.get(key), str) and row[key]
            for key in ("pane_id", "terminal_id")
        )
        and all(
            row.get(key) is None or isinstance(row[key], str)
            for key in ("agent", "display_agent")
        )
        and isinstance(tokens, dict)
        and all(isinstance(value, str) for value in tokens.values())
    )


def _resolved_socket(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        raise HerdrError("cannot resolve a Herdr server socket") from None


class Herdr:
    def __init__(self, env: Mapping[str, str], session: str | None = None) -> None:
        self.env = dict(env)
        self.session = session
        self.prefix = ["herdr"]
        if session:
            self.prefix.extend(["--session", session])

    def call(self, *args: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [*self.prefix, *args],
                env=self.env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HerdrError("could not run the Herdr CLI") from exc
        except UnicodeDecodeError as exc:
            raise HerdrError("Herdr returned a response with invalid encoding") from exc
        if result.returncode:
            # CLI stderr can contain local session references. Keep it private.
            raise HerdrError(
                f"Herdr {args[0]} {args[1]} failed (exit {result.returncode})"
            )
        if not result.stdout.strip():
            return {}
        try:
            value = json.loads(result.stdout)
        except (ValueError, RecursionError):
            raise HerdrError("Herdr returned invalid JSON") from None
        if not isinstance(value, dict) or "error" in value:
            raise HerdrError("Herdr returned an invalid/error response")
        data = value.get("result", value)
        if not isinstance(data, dict):
            raise HerdrError("Herdr response result is not an object")
        return data

    def running_sessions(self) -> list[dict[str, Any]]:
        sessions = self.call("session", "list", "--json").get("sessions")
        if not isinstance(sessions, list):
            raise HerdrError("invalid Herdr server list")
        running = []
        for row in sessions:
            if not isinstance(row, dict) or type(row.get("running")) is not bool:
                raise HerdrError("invalid Herdr server entry")
            if not row["running"]:
                continue
            socket, name = row.get("socket_path"), row.get("name")
            if (
                not isinstance(socket, str)
                or not Path(socket).is_absolute()
                or not isinstance(name, str)
                or not name
                or type(row.get("default", False)) is not bool
            ):
                raise HerdrError("invalid Herdr server identity")
            running.append({**row, "socket_path": _resolved_socket(socket)})
        return running

    def socket_path(self) -> str:
        if self.session is None and self.env.get("HERDR_SOCKET_PATH"):
            return _resolved_socket(self.env["HERDR_SOCKET_PATH"])
        selected = [
            row
            for row in self.running_sessions()
            if (row.get("name") == self.session if self.session else row.get("default"))
        ]
        if len(selected) != 1:
            raise HerdrError("cannot resolve the selected Herdr server socket")
        return str(selected[0]["socket_path"])

    def pane(self, pane_id: str | None = None) -> dict[str, Any]:
        result = (
            self.call("pane", "get", pane_id)
            if pane_id
            else self.call("pane", "current", "--current")
        )
        pane = result.get("pane")
        if not _valid_pane(pane):
            raise HerdrError("Herdr returned invalid pane identity or metadata")
        return pane

    def agents(self) -> list[dict[str, Any]]:
        agents = self.call("agent", "list").get("agents")
        if not isinstance(agents, list) or not all(_valid_pane(row) for row in agents):
            raise HerdrError("invalid Herdr agent list")
        return agents

    def foreground_processes(self, pane_id: str) -> list[dict[str, Any]]:
        info = self.call("pane", "process-info", "--pane", pane_id).get("process_info")
        processes = info.get("foreground_processes") if isinstance(info, dict) else None
        if not isinstance(processes, list) or not all(
            isinstance(proc, dict)
            and isinstance(proc.get("name"), str)
            and type(proc.get("pid")) is int
            and 0 < proc["pid"] <= 0xFFFF_FFFF
            for proc in processes
        ):
            raise HerdrError("invalid Herdr foreground process information")
        return processes

    def publish(
        self,
        pane: dict[str, Any],
        *,
        agent: str | None,
        display: str | None,
        tokens: dict[str, str],
        validate: Callable[[], None] | None = None,
        source: str = "tennis-parent",
        tokens_only: bool = False,
    ) -> dict[str, Any]:
        """Recheck the occupant, patch our fields, and verify the actual readback."""
        current = self.pane(pane["pane_id"])
        if (current["terminal_id"], current.get("agent")) != (
            pane["terminal_id"],
            pane.get("agent"),
        ):
            raise ValueError("Herdr pane occupant changed before metadata update")
        if current.get("display_agent") != pane.get("display_agent") or any(
            current.get("tokens", {}).get(k) != pane.get("tokens", {}).get(k)
            for k in tokens
        ):
            raise ValueError("Herdr display ownership changed during preparation")
        if validate:
            validate()
        existing = current.get("tokens", {})
        other_tokens = {
            key: value for key, value in existing.items() if key not in tokens
        }
        fields_match = all(existing.get(k, "") == value for k, value in tokens.items())
        if current.get("display_agent") == display and fields_match:
            return {"status": "unchanged", "pane": pane["pane_id"]}
        args = [
            "pane",
            "report-metadata",
            pane["pane_id"],
            "--source",
            source,
        ]
        if not tokens_only:
            if agent is not None:
                args.extend(["--agent", agent])
            args.extend(
                ["--display-agent", display] if display else ["--clear-display-agent"]
            )
        for key, value in tokens.items():
            args.extend(
                ["--token", f"{key}={value}"] if value else ["--clear-token", key]
            )
        self.call(*args)
        after = self.pane(pane["pane_id"])
        if validate:
            validate()
        if (after["terminal_id"], after.get("agent")) != (pane["terminal_id"], agent):
            raise ValueError("Herdr pane changed during metadata update")
        if (
            after.get("display_agent") != display
            or any(
                after.get("tokens", {}).get(k, "") != value
                for k, value in tokens.items()
            )
            or {
                key: value
                for key, value in after.get("tokens", {}).items()
                if key not in tokens
            }
            != other_tokens
        ):
            raise ValueError(
                "Herdr metadata readback does not match the requested label"
            )
        return {"status": "updated", "pane": pane["pane_id"]}
