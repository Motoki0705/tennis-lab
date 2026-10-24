"""Small, explicitly targeted Herdr CLI adapter."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class Herdr:
    def __init__(self, env: Mapping[str, str], session: str | None = None) -> None:
        self.env = dict(env)
        self.session = session
        self.prefix = ["herdr"]
        if session:
            self.prefix.extend(["--session", session])

    def call(self, *args: str) -> dict[str, Any]:
        result = subprocess.run(
            [*self.prefix, *args],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode:
            # CLI stderr can contain local session references. Keep it private.
            raise ValueError(
                f"Herdr {args[0]} {args[1]} failed (exit {result.returncode})"
            )
        if not result.stdout.strip():
            return {}
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or "error" in value:
            raise ValueError("Herdr returned an invalid/error response")
        data = value.get("result", value)
        if not isinstance(data, dict):
            raise ValueError("Herdr response result is not an object")
        return data

    def socket_path(self) -> str:
        if self.session is None and self.env.get("HERDR_SOCKET_PATH"):
            return str(Path(self.env["HERDR_SOCKET_PATH"]).resolve())
        sessions = self.call("session", "list", "--json").get("sessions", [])
        selected = [
            row
            for row in sessions
            if (row.get("name") == self.session if self.session else row.get("default"))
            and row.get("running")
        ]
        if len(selected) != 1 or not selected[0].get("socket_path"):
            raise ValueError("cannot resolve the selected Herdr server socket")
        return str(Path(selected[0]["socket_path"]).resolve())

    def pane(self, pane_id: str | None = None) -> dict[str, Any]:
        result = (
            self.call("pane", "get", pane_id)
            if pane_id
            else self.call("pane", "current", "--current")
        )
        pane = result.get("pane")
        if not isinstance(pane, dict) or not all(
            isinstance(pane.get(k), str) and pane[k] for k in ("pane_id", "terminal_id")
        ):
            raise ValueError("Herdr did not return a valid pane identity")
        return pane

    def publish(
        self,
        pane: dict[str, Any],
        *,
        agent: str,
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
