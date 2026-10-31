"""Read-only Herdr runtime overview, separate from board membership."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from herdr_labels.client import Herdr

from .errors import BoardError
from .safety import public
from .service import STATES


def snapshot(env: Mapping[str, str]) -> dict[str, Any]:
    """Observe all running local Herdr servers without assigning board identities."""
    try:
        servers = Herdr(env).call("session", "list", "--json")["sessions"]
        if not isinstance(servers, list):
            raise ValueError("invalid server list")
        rows = []
        for server in servers:
            if not isinstance(server, dict) or type(server.get("running")) is not bool:
                raise ValueError("invalid server")
            if not server["running"]:
                continue
            if not isinstance(server.get("name"), str) or not server["name"]:
                raise ValueError("invalid server name")
            agents = Herdr(env, server["name"]).call("agent", "list")["agents"]
            if not isinstance(agents, list):
                raise ValueError("invalid agent list")
            rows.extend(agents)
        private = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid agent")
            for key in ("pane_id", "terminal_id", "cwd", "foreground_cwd"):
                if isinstance(row.get(key), str):
                    private.append(row[key])
            native = row.get("agent_session")
            if isinstance(native, dict) and isinstance(native.get("value"), str):
                private.append(native["value"])
        secrets = {
            str(i): {"session": f"runtime:{value}"} for i, value in enumerate(private)
        }
        counts: Counter[str] = Counter()
        result = []
        for row in rows:
            engine = row.get("agent")
            status = row.get("agent_status", row.get("state"))
            if not isinstance(engine, str) or not re.fullmatch(
                r"[a-zA-Z0-9_-]{1,40}", engine
            ):
                raise ValueError("invalid agent engine")
            if status not in STATES | {"unknown"}:
                raise ValueError("invalid agent state")
            title = row.get("terminal_title_stripped", "")
            if not isinstance(title, str):
                raise ValueError("invalid agent title")
            title = title[:200]
            title_status = "available"
            try:
                public(title, secrets, env)
            except BoardError:
                title, title_status = "", "redacted"
            counts[engine] += 1
            result.append(
                {
                    "label": f"{engine} {counts[engine]}",
                    "engine": engine,
                    "state": status,
                    "title": title,
                    "title_status": title_status,
                }
            )
        public(result, secrets, env)
        return {"agents": result, "observed_at": datetime.now(UTC).isoformat()}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        raise BoardError("Herdr runtime overview unavailable or inconsistent") from None
