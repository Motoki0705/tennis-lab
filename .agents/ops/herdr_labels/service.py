"""Ownership projection; the parent registry remains the only assignment authority."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from issues import parse_title
from parent.github import GitHub
from parent.state import Store, session_identity

from .client import Herdr
from .coordination import Guard, owner_token, process_stamp, state_root

TOKEN_KEYS = ("tl_parent", "tl_topic", "tl_engine", "tl_owner")


def short_topic(text: str, columns: int = 20) -> str:
    if (
        not text
        or text != text.strip()
        or any(unicodedata.category(c).startswith("C") for c in text)
    ):
        raise ValueError("topic must be nonempty text without control characters")
    used = 0
    for index, char in enumerate(text):
        width = (
            0
            if unicodedata.combining(char)
            else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
        )
        if used + width > columns:
            return short_topic(text[:index], columns - 1) + "…"
        used += width
    return text


def registry(env: Mapping[str, str]) -> dict[str, Any]:
    """Read the atomically replaced registry, without taking its long-held writer lock."""
    path = state_root(env) / "parent-sessions/registry.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("invalid parent registry version")
    parents = data.get("parents")
    if not isinstance(parents, dict):
        raise ValueError("invalid parent registry records")
    normalized: dict[str, Any] = {}
    owners: set[str] = set()
    for key, row in parents.items():
        if not isinstance(key, str) or key.casefold() in normalized:
            raise ValueError("ambiguous parent registry key")
        if not isinstance(row, dict) or not isinstance(row.get("session"), str):
            raise ValueError("invalid parent registry session")
        if row.get("status") not in {"active", "handoff_pending", "released", "ended"}:
            raise ValueError("invalid parent registry assignment")
        if row["status"] in {"active", "handoff_pending"}:
            if row["session"] in owners:
                raise ValueError("multiple parents claim the same session")
            owners.add(row["session"])
        normalized[key.casefold()] = row
    return normalized


def binding_path(env: Mapping[str, str], identity: str) -> Path:
    return state_root(env) / "herdr-labels/bindings" / f"{owner_token(identity)}.json"


def bind(
    env: Mapping[str, str], session: str, pane_id: str, terminal_id: str
) -> dict[str, Any]:
    """Explicit setup for a tool runner that does not inherit Herdr pane variables."""
    identity = session_identity(dict(env))
    client = Herdr(env, session)
    pane = client.pane(pane_id)
    if (
        pane["terminal_id"] != terminal_id
        or pane.get("agent") != identity.split(":", 1)[0]
    ):
        raise ValueError("binding target identity does not match")
    path = binding_path(env, identity)
    previous = json.loads(path.read_text()) if path.exists() else None
    record = {
        "version": 1,
        "herdr_session": session,
        "pane_id": pane_id,
        "terminal_id": terminal_id,
        "processes": process_stamp(client, pane),
    }
    if previous and previous != record:
        raise ValueError(
            "a different binding already exists; inspect and explicitly remove it"
        )
    with Guard(env, identity, client, pane).hold(start=True, confirmed=True):
        Store.write_json(path, record)
    return {"status": "bound", "pane": pane_id}


def target(
    env: Mapping[str, str], identity: str
) -> tuple[Herdr, dict[str, Any]] | None:
    if env.get("HERDR_ENV") == "1":
        if not env.get("HERDR_PANE_ID"):
            raise ValueError("HERDR_PANE_ID is missing in a Herdr environment")
        client = Herdr(env)
        # --current resolves the caller's pane even after it has moved workspaces.
        pane = client.pane()
    else:
        path = binding_path(env, identity)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError("invalid Herdr binding version")
        client = Herdr(env, data["herdr_session"])
        pane = client.pane(data["pane_id"])
        if (
            pane["terminal_id"] != data["terminal_id"]
            or process_stamp(client, pane) != data["processes"]
        ):
            raise ValueError(
                "bound Herdr agent has exited or been replaced; rebind explicitly"
            )
    if pane.get("agent") != identity.split(":", 1)[0]:
        raise ValueError("the target pane does not contain the caller's agent kind")
    return client, pane


def sync(
    env: Mapping[str, str],
    *,
    topic: str | None = None,
    expected_parent: int | None = None,
    clear: bool = False,
    session_start: bool = False,
    session_end: bool = False,
    confirm_parent: bool = False,
) -> dict[str, Any]:
    identity = session_identity(dict(env))
    destination = target(env, identity)
    if destination is None:
        return {
            "status": "not_configured",
            "reason": "no Herdr caller context or explicit binding",
        }
    client, pane = destination
    if (session_start or session_end) and env.get("HERDR_ENV") != "1":
        raise ValueError("lifecycle events require the inherited Herdr caller context")
    guard = Guard(env, identity, client, pane)
    if session_start or session_end:
        guard.require_native_ancestor()
    with guard.hold(start=session_start, end=session_end):
        return _sync(
            env,
            identity,
            client,
            guard.pane,
            guard,
            topic=topic,
            expected_parent=expected_parent,
            clear=clear,
            session_start=session_start,
            confirm_parent=confirm_parent,
        )


def _sync(
    env: Mapping[str, str],
    identity: str,
    client: Herdr,
    pane: dict[str, Any],
    guard: Guard,
    *,
    topic: str | None,
    expected_parent: int | None,
    clear: bool,
    session_start: bool,
    confirm_parent: bool,
) -> dict[str, Any]:
    token = owner_token(identity)
    snapshot = registry(env)
    active = [
        (key, row)
        for key, row in snapshot.items()
        if row["session"] == identity and row["status"] == "active"
    ]
    existing = pane.get("tokens", {})
    previous_owner = existing.get("tl_owner")
    # Only a runtime SessionStart event can discard the predecessor's label
    # in its own inherited pane. Ordinary sync/clear must fail closed.
    if (
        previous_owner
        and previous_owner != token
        and (not session_start or env.get("HERDR_ENV") != "1")
    ):
        raise ValueError("label belongs to another session; refusing to replace it")
    if clear or not active:
        if expected_parent is not None and not clear:
            raise ValueError("the caller does not own the requested active parent")
        if not previous_owner:
            return {"status": "unassigned", "pane": pane["pane_id"]}
        if registry(env) != snapshot:
            raise ValueError("parent registry changed before label cleanup; retry sync")
        return client.publish(
            pane,
            agent=identity.split(":", 1)[0],
            display=None,
            tokens={key: "" for key in TOKEN_KEYS},
            validate=guard.check,
        )
    key, row = active[0]
    repo, separator, number = key.rpartition("#")
    if not separator or not number.isdecimal() or int(number) < 1 or "/" not in repo:
        raise ValueError("invalid parent registry key")
    if expected_parent is not None and int(number) != expected_parent:
        raise ValueError("requested parent differs from the caller's registered parent")
    agents = client.call("agent", "list").get("agents")
    if not isinstance(agents, list) or not all(isinstance(p, dict) for p in agents):
        raise ValueError("Herdr returned an invalid agent list")
    if any(
        p.get("terminal_id") != pane["terminal_id"]
        and p.get("agent") == identity.split(":", 1)[0]
        and p.get("tokens", {}).get("tl_owner") == token
        for p in agents
    ):
        raise ValueError(
            "the same session already labels another live pane; "
            "inspect duplicated sessions before synchronizing"
        )
    guard.claim(allow_create=confirm_parent)
    if topic is None:
        if existing.get("tl_owner") == token and existing.get("tl_parent") == key:
            topic = existing.get("tl_topic")
        if not topic:
            issue = GitHub(repo).issue(int(number))
            if issue["state"] != "open":
                raise ValueError(
                    "registered parent is closed; release ownership before relabelling"
                )
            parsed = parse_title(issue["title"])
            if not parsed.identifier:
                raise ValueError(
                    "parent title has no short identifier; pass --topic explicitly"
                )
            topic = parsed.identifier
    label = short_topic(topic)
    # Do not project an assignment that changed while fetching its issue title.
    if registry(env) != snapshot:
        raise ValueError("parent registry changed during label preparation; retry sync")
    result = client.publish(
        pane,
        agent=identity.split(":", 1)[0],
        display=f"#{number} · 親",
        tokens={
            "tl_parent": key,
            "tl_topic": label,
            "tl_engine": identity.split(":", 1)[0],
            "tl_owner": token,
        },
        validate=guard.check,
    )
    return {**result, "parent": int(number), "topic": label}
