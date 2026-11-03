"""Read-only Herdr runtime observations, separate from board membership."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from herdr_labels.client import Herdr
from herdr_labels.coordination import owner_token, read_record, state_root
from herdr_labels.service import registry

from .errors import BoardError
from .safety import public
from .service import STATES, _runtime_state

PRIVATE_KEYS = ("pane_id", "terminal_id", "cwd", "foreground_cwd")
LIVE = {"active", "handoff_pending"}


@dataclass
class Server:
    name: str
    scope: str
    client: Herdr
    agents: list[dict[str, Any]]


def _servers(env: Mapping[str, str]) -> list[Server]:
    """All running local Herdr servers and their agent lists, in Herdr order."""
    servers = Herdr(env).call("session", "list", "--json")["sessions"]
    if not isinstance(servers, list):
        raise ValueError("invalid server list")
    result = []
    for server in servers:
        if not isinstance(server, dict) or type(server.get("running")) is not bool:
            raise ValueError("invalid server")
        if not server["running"]:
            continue
        if not isinstance(server.get("name"), str) or not server["name"]:
            raise ValueError("invalid server name")
        if not isinstance(server.get("socket_path"), str) or not server["socket_path"]:
            raise ValueError("invalid server socket")
        client = Herdr(env, server["name"])
        agents = client.call("agent", "list")["agents"]
        if not isinstance(agents, list) or not all(isinstance(a, dict) for a in agents):
            raise ValueError("invalid agent list")
        scope = owner_token(str(Path(server["socket_path"]).resolve()))
        result.append(Server(server["name"], scope, client, agents))
    return result


def _private(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Runtime identities in the shape `public` reads registry sessions from."""
    values = []
    for row in rows:
        for key in PRIVATE_KEYS:
            if isinstance(row.get(key), str):
                values.append(row[key])
        native = row.get("agent_session")
        if isinstance(native, dict) and isinstance(native.get("value"), str):
            values.append(native["value"])
    return {str(i): {"session": f"runtime:{value}"} for i, value in enumerate(values)}


def _project(
    rows: list[dict[str, Any]], secrets: Mapping[str, Any], env: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Public agent rows. Titles with private values are redacted, never shown."""
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
    return result


def snapshot(env: Mapping[str, str]) -> dict[str, Any]:
    """Observe all running local Herdr servers without assigning board identities."""
    try:
        rows = [agent for server in _servers(env) for agent in server.agents]
        secrets = _private(rows)
        result = _project(rows, secrets, env)
        public(result, secrets, env)
        return {"agents": result, "observed_at": datetime.now(UTC).isoformat()}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        raise BoardError("Herdr runtime overview unavailable or inconsistent") from None


def _native(row: dict[str, Any], engine: str, session: str) -> bool:
    """Herdr's own conversation reference names exactly this registered session."""
    native = row.get("agent_session")
    return (
        isinstance(native, dict)
        and native.get("kind") == "id"
        and native.get("value") == session
        and row.get("agent") == engine
        and native.get("agent", engine) == engine
    )


def sessions(env: Mapping[str, str]) -> dict[str, Any]:
    """Herdr agents for the execution graph, joined to live parents by identity only.

    A parent joins one agent when its confirmed owner record (herdr_labels) or
    Herdr's native session reference identifies exactly one terminal. The same
    session open in several terminals, disagreeing sources, or one agent claimed
    by several parents are reported as ambiguous instead of choosing one. Titles,
    focus and working directories are never used for joining.

    The result is private to the server process: `parents[*].index` points into
    `sessions`, whose rows are public. Keys are one-way digests of Herdr scope and
    terminal, stable across observations for UI selection.
    """
    try:
        parents = registry(env)
        servers = _servers(env)
        located = [(server, agent) for server in servers for agent in server.agents]
        rows = [agent for _, agent in located]
        secrets = {**parents, **_private(rows)}
        public_rows = _project(rows, secrets, env)
        for (server, agent), row in zip(located, public_rows, strict=True):
            terminal = agent.get("terminal_id")
            if not isinstance(terminal, str) or not terminal:
                raise ValueError("invalid agent terminal")
            digest = hashlib.sha256(f"{server.scope}:{terminal}".encode()).hexdigest()
            row["key"] = "herdr-" + digest[:12]
        if len({row["key"] for row in public_rows}) != len(public_rows):
            raise ValueError("duplicate Herdr terminal")
        root = state_root(env) / "herdr-labels"
        matches: dict[str, dict[str, Any]] = {}
        for key, row in sorted(parents.items()):
            if row["status"] not in LIVE:
                continue
            identity = row["session"]
            engine, _, session = identity.partition(":")
            owned: set[int] = set()
            record = read_record(root / "owners" / f"{owner_token(identity)}.json")
            if (
                record is not None
                and isinstance(record.get("scope"), str)
                and isinstance(record.get("terminal_id"), str)
            ):
                for index, (server, agent) in enumerate(located):
                    if (
                        server.scope == record["scope"]
                        and agent.get("terminal_id") == record["terminal_id"]
                        and agent.get("agent") == engine
                        and _runtime_state(
                            root,
                            server.scope,
                            identity,
                            record,
                            server.agents,
                            server.client,
                        )
                        != "unknown"
                    ):
                        owned.add(index)
            opened = {
                index
                for index, agent in enumerate(rows)
                if _native(agent, engine, session)
            }
            found = owned | opened
            if (
                len(opened) > 1
                or len(owned) > 1
                or (owned and opened and owned != opened)
            ):
                status, method, index_ = "ambiguous", None, None
            elif found:
                status, index_ = "matched", next(iter(found))
                method = (
                    "owner_native"
                    if owned and opened
                    else "owner"
                    if owned
                    else "native"
                )
            else:
                status, method, index_ = "absent", None, None
            matches[key] = {
                "status": status,
                "method": method,
                "index": index_,
                "generation": row["generation"],
                "assignment": row["status"],
            }
        claimed = Counter(
            m["index"] for m in matches.values() if m["index"] is not None
        )
        for match in matches.values():
            if match["index"] is not None and claimed[match["index"]] > 1:
                match.update(status="ambiguous", method=None, index=None)
        if registry(env) != parents:
            raise BoardError("parent registry changed during Herdr lookup; retry")
        public(public_rows, secrets, env)
        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "sessions": public_rows,
            "parents": matches,
        }
    except BoardError:
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        raise BoardError("Herdr sessions unavailable or inconsistent") from None
