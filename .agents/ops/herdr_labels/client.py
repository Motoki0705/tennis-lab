"""Small, explicitly targeted Herdr CLI adapter."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from typing import Any


class Herdr:
    def __init__(self, env: Mapping[str, str], session: str | None = None) -> None:
        self.env = dict(env)
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
    ) -> dict[str, Any]:
        """Recheck the occupant, patch our fields, and verify the actual readback."""
        current = self.pane(pane["pane_id"])
        if (current["terminal_id"], current.get("agent")) != (
            pane["terminal_id"],
            pane.get("agent"),
        ):
            raise ValueError("Herdr pane occupant changed before metadata update")
        existing = current.get("tokens", {})
        fields_match = all(existing.get(k, "") == value for k, value in tokens.items())
        if current.get("display_agent") == display and fields_match:
            return {"status": "unchanged", "pane": pane["pane_id"]}
        args = [
            "pane",
            "report-metadata",
            pane["pane_id"],
            "--source",
            "tennis-parent",
            "--agent",
            agent,
        ]
        args.extend(
            ["--display-agent", display] if display else ["--clear-display-agent"]
        )
        for key, value in tokens.items():
            args.extend(
                ["--token", f"{key}={value}"] if value else ["--clear-token", key]
            )
        self.call(*args)
        after = self.pane(pane["pane_id"])
        if (after["terminal_id"], after.get("agent")) != (pane["terminal_id"], agent):
            raise ValueError("Herdr pane changed during metadata update")
        if after.get("display_agent") != display or any(
            after.get("tokens", {}).get(k, "") != value for k, value in tokens.items()
        ):
            raise ValueError(
                "Herdr metadata readback does not match the requested label"
            )
        return {"status": "updated", "pane": pane["pane_id"]}
