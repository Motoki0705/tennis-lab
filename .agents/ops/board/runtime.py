"""Read-only Herdr runtime observations, separate from board membership."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from herdr_labels.client import Herdr
from herdr_labels.coordination import (
    ForegroundAgentUnavailable,
    owner_token,
    process_stamp,
    read_record,
    state_root,
)
from herdr_labels.service import registry
from lib.agent_events import parse_json
from lib.agy_session import AgyTui
from worker_status.records import claude_messages, tail, timed_text, timestamp
from worker_status.safety import Guard

from .errors import BoardError
from .safety import public, truncate
from .service import STATES, _runtime_state

PRIVATE_KEYS = ("pane_id", "terminal_id", "cwd", "foreground_cwd")
LIVE = {"active", "handoff_pending"}
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


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


def _private(
    rows: list[dict[str, Any]], session_ids: tuple[str, ...] = ()
) -> dict[str, dict[str, str]]:
    """Runtime identities in the shape `public` reads registry sessions from."""
    values = list(session_ids)
    for row in rows:
        for key in PRIVATE_KEYS:
            if isinstance(row.get(key), str):
                values.append(row[key])
        native = row.get("agent_session")
        if isinstance(native, dict) and isinstance(native.get("value"), str):
            values.append(native["value"])
    return {str(i): {"session": f"runtime:{value}"} for i, value in enumerate(values)}


def _transcript_path(
    engine: str, session_id: str, env: Mapping[str, str]
) -> Path | None:
    home = Path(env["HOME"]) if env.get("HOME") else Path.home()
    if engine == "agy":
        return (
            home
            / ".gemini/antigravity-cli/brain"
            / session_id
            / ".system_generated/logs/transcript.jsonl"
        )
    if engine == "claude":
        root = Path(env.get("CLAUDE_CONFIG_DIR") or home / ".claude") / "projects"
        matches = [p for p in root.glob(f"*/{session_id}.jsonl") if p.is_file()]
    else:
        root = Path(env.get("CODEX_HOME") or home / ".codex") / "sessions"
        matches = [
            p
            for p in root.rglob(f"*{session_id}.jsonl")
            if (p.stem == session_id or p.stem.endswith(f"-{session_id}"))
            and p.is_file()
        ]
    return matches[0] if len(matches) == 1 else None


def _block_text(content: Any, kind: str) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == kind
        and isinstance(block.get("text"), str)
    ]


def _codex_messages(event: dict[str, Any]) -> tuple[str | None, list[str]]:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None, []
    item = payload
    parts: list[str] = []
    if event.get("type") == "response_item":
        if item.get("type") == "message" and item.get("role") == "assistant":
            parts = _block_text(item.get("content"), "output_text")
        elif item.get("type") == "reasoning":
            parts = _block_text(item.get("summary"), "summary_text")
    elif event.get("type") == "event_msg" and payload.get("type") == "item_completed":
        completed = payload.get("item")
        if not isinstance(completed, dict):
            return None, []
        item = completed
        if item.get("type") in ("AgentMessage", "agent_message"):
            text = item.get("text")
            parts = (
                [text]
                if isinstance(text, str)
                else (
                    _block_text(item.get("content"), "Text")
                    or _block_text(item.get("content"), "output_text")
                )
            )
        elif item.get("type") == "Reasoning" and isinstance(
            item.get("summary_text"), list
        ):
            parts = [text for text in item["summary_text"] if isinstance(text, str)]
    item_id = item.get("id")
    return (item_id if isinstance(item_id, str) else None), parts


def _agy_messages(event: dict[str, Any]) -> list[str]:
    if event.get("type") == "PLANNER_RESPONSE":
        # The local CLI transcript flattens the planner's text fields.
        values = (event.get("thinking"), event.get("content"))
    elif (
        event.get("type", event.get("stepType")) == "CORTEX_STEP_TYPE_PLANNER_RESPONSE"
    ):
        planner = event.get("plannerResponse")
        if not isinstance(planner, dict):
            return []
        values = (planner.get("thinking"), planner.get("response"))
    else:
        return []
    return [value for value in values if isinstance(value, str)]


def session_summary(
    engine: str | None,
    session_id: str | None,
    guard: Guard,
    env: Mapping[str, str],
    *,
    tail_bytes: int = 262144,
    limit: int = 20,
) -> dict[str, Any]:
    """Publish a bounded chronological tail of the assistant's text and summaries."""
    empty: dict[str, Any] = {"messages": [], "tail_truncated": False}
    if (
        engine not in {"agy", "claude", "codex"}
        or not session_id
        or not SESSION_ID.fullmatch(session_id)
        or limit < 1
        or tail_bytes < 1
    ):
        return empty
    guard.remember(session_id)
    try:
        path = _transcript_path(engine, session_id, env)
        if path is None or not path.is_file():
            return empty
        guard.remember_path(str(path))
        lines, truncated, _, _ = tail(path, tail_bytes)
    except OSError:
        return empty
    messages: list[tuple[str, str | None]] = []
    seen_ids: set[str] = set()
    for _, _, raw in lines:
        try:
            event = parse_json(raw)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        if engine == "codex":
            item_id, parts = _codex_messages(event)
            if item_id and any(text.strip() for text in parts):
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
        elif engine == "claude":
            parts = [text for _, text in claude_messages(event)]
        else:
            parts = _agy_messages(event)
        when = None
        with suppress(ValueError, OverflowError):
            when = timestamp(event.get("timestamp", event.get("created_at")))
        for text in parts:
            if text.strip() and (not messages or messages[-1][0] != text):
                messages.append((text, when))
    return {
        "messages": [timed_text(guard, text, when) for text, when in messages[-limit:]],
        "tail_truncated": truncated or len(messages) > limit,
    }


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
        title_status = "available"
        try:
            public(title, secrets, env)
            shortened = truncate(title, 200)
            if shortened != title:
                public(shortened, secrets, env)
            title = shortened
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


def _native_session(row: dict[str, Any]) -> str | None:
    if row.get("agent") == "agy":
        return None  # agy conversation identity comes from its foreground TUI lock.
    native = row.get("agent_session")
    if (
        isinstance(native, dict)
        and native.get("kind") == "id"
        and native.get("agent", row["agent"]) == row["agent"]
        and isinstance(native.get("value"), str)
        and SESSION_ID.fullmatch(native["value"])
    ):
        return str(native["value"])
    return None


def _agy_conversation(server: Server, agent: dict[str, Any]) -> str | None:
    """Read a conversation lock only from the pane's unique foreground agy TUI."""
    if not agent.get("pane_id"):
        return None
    try:
        (stamp,) = process_stamp(server.client, agent)
    except (ForegroundAgentUnavailable, FileNotFoundError, ProcessLookupError):
        return None
    with suppress(OSError, ValueError):
        session_id = AgyTui(stamp["pid"], stamp["start"]).conversation()
        if isinstance(session_id, str):
            return session_id
    return None


def sessions(env: Mapping[str, str]) -> dict[str, Any]:
    """Herdr agents for the execution graph, joined to live parents by identity only.

    A parent joins one agent when its confirmed owner record (herdr_labels) or
    Herdr's native session reference identifies exactly one terminal. The same
    session open in several terminals, disagreeing sources, or one agent claimed
    by several parents are reported as ambiguous instead of choosing one. Titles,
    focus and working directories are never used for joining.

    The result is private to the server process: `parents[*].index` points into
    `sessions`, whose rows are public. Keys are one-way digests of Herdr scope and
    terminal, stable across observations for UI selection. Conversation references
    and redaction inputs stay here so local polling can read new transcript lines.
    """
    try:
        parents = registry(env)
        servers = _servers(env)
        located = [(server, agent) for server in servers for agent in server.agents]
        rows = [agent for _, agent in located]
        native = [_native_session(row) for row in rows]
        keys = []
        for server, agent in located:
            terminal = agent.get("terminal_id")
            if not isinstance(terminal, str) or not terminal:
                raise ValueError("invalid agent terminal")
            digest = hashlib.sha256(f"{server.scope}:{terminal}".encode()).hexdigest()
            keys.append("herdr-" + digest[:12])
        if len(set(keys)) != len(keys):
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
                for index, native_id in enumerate(native)
                if native_id == session and rows[index]["agent"] == engine
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
        session_ids = {
            row_key: session_id
            for row_key, session_id in zip(keys, native, strict=True)
            if session_id is not None
        }
        for key, match in matches.items():
            if match["index"] is not None:
                row_key = keys[match["index"]]
                session_ids[row_key] = parents[key]["session"].partition(":")[2]
        for (server, agent), row_key in zip(located, keys, strict=True):
            if agent.get("agent") != "agy" or row_key in session_ids:
                continue
            session_id = _agy_conversation(server, agent)
            if session_id:
                session_ids[row_key] = session_id
        private = _private(rows, tuple(session_ids.values()))
        secrets = {**parents, **private}
        if registry(env) != parents:
            raise BoardError("parent registry changed during Herdr lookup; retry")
        public_rows = _project(rows, secrets, env)
        for row, row_key in zip(public_rows, keys, strict=True):
            row["key"] = row_key
        public(public_rows, secrets, env)
        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "sessions": public_rows,
            "parents": matches,
            "session_ids": session_ids,
            "private": private,
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
