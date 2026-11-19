"""Ownership projection; the parent registry remains the only assignment authority."""

from __future__ import annotations

import json
import subprocess
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lib.identity import session_identity
from parent.policy import digest
from parent.state import Store

from .client import Herdr
from .coordination import Guard, owner_token, process_stamp, state_root

TOKEN_KEYS = (
    "tl_name",
    "tl_parent",
    "tl_issue",
    "tl_topic",
    "tl_engine",
    "tl_owner",
)
INBOX_KEYS = ("tl_inbox", "tl_inbox_label")


def issue_label(parent: int) -> str:
    """Short sidebar cell; ``tl_parent`` keeps the full ``owner/repo#N`` key."""
    return f"#{parent}"


class VerifiedSessionStartError(ValueError):
    """Claude display sync failed after native caller verification succeeded."""


def board_name(env: Mapping[str, str], repo: str, parent: int) -> str | None:
    """Read an existing live assignment; display sync never allocates a name."""
    from board.names import canonical_name, for_parent
    from board.storage import Store as BoardStore
    from board.storage import directory

    store = BoardStore(directory(env))
    if not store.path.exists() and not store.path.is_symlink():
        return None
    with store.read() as connection:
        assignment = for_parent(connection, repo, parent)
        if assignment is None or assignment.retired_at is not None:
            return None
        return canonical_name(assignment.name)


def clear_inbox(client: Herdr, pane: dict[str, Any], guard: Guard) -> dict[str, Any]:
    """Lifecycle cleanup uses the board source while holding the pane guard."""
    if any(key in pane.get("tokens", {}) for key in INBOX_KEYS):
        client.publish(
            pane,
            agent=pane["agent"],
            display=pane.get("display_agent"),
            tokens={key: "" for key in INBOX_KEYS},
            validate=guard.check,
            source="tennis-board",
            tokens_only=True,
        )
        return client.pane(pane["pane_id"])
    return pane


SUMMARY_COLUMNS = 24


def display_columns(text: str) -> int:
    return sum(
        0
        if unicodedata.combining(char)
        else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
        for char in text
    )


def validate_summary(text: str) -> str:
    """The parent writes the sidebar summary; never truncate or rewrite it."""
    if (
        not isinstance(text, str)
        or not text
        or text != text.strip()
        or any(unicodedata.category(c).startswith("C") for c in text)
    ):
        raise ValueError("summary must be nonempty text without control characters")
    columns = display_columns(text)
    if columns > SUMMARY_COLUMNS:
        raise ValueError(
            f"summary is {columns} display columns; shorten it to {SUMMARY_COLUMNS} "
            "or fewer (full-width characters count as 2)"
        )
    return text


def parent_summary(env: Mapping[str, str], key: str, owner: Mapping[str, Any]) -> str:
    """Read the summary saved with the parent record matching the ownership index."""
    if "tasks" in owner:  # version 1 keeps the full record in the index
        record: Mapping[str, Any] = owner
    else:
        path = state_root(env) / "parent-sessions/parents" / (digest(key) + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or any(
            record.get(field) != owner.get(field)
            for field in ("session", "status", "generation")
        ):
            raise ValueError("parent record disagrees with its index; retry sync")
    value = record.get("display_summary")
    if value is None:
        number = key.rpartition("#")[2]
        raise ValueError(
            f"parent #{number} has no sidebar summary; register one with "
            f"`python -m parent --parent {number} summary --text <summary>`"
        )
    return validate_summary(value)


def registry(env: Mapping[str, str]) -> dict[str, Any]:
    """Read the atomically replaced registry, without taking its long-held writer lock."""
    path = state_root(env) / "parent-sessions/registry.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") not in {1, 2}:
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
    expected_parent: int | None = None,
    clear: bool = False,
    session_start: bool = False,
    session_end: bool = False,
    confirm_parent: bool = False,
    bind_native: bool = False,
) -> dict[str, Any]:
    if bind_native and (not session_start or session_end):
        raise ValueError("native binding requires a SessionStart event")
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
    try:
        with guard.hold(start=session_start, end=session_end):
            result = _sync(
                env,
                identity,
                client,
                guard.pane,
                guard,
                expected_parent=expected_parent,
                clear=clear,
                session_start=session_start,
                confirm_parent=confirm_parent,
            )
            if bind_native:
                _bind_native(env, identity, client, guard)
            return result
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        if session_start and identity.startswith("claude:") and guard.native_verified:
            raise VerifiedSessionStartError(str(exc)) from exc
        raise


def _bind_native(
    env: Mapping[str, str], identity: str, client: Herdr, guard: Guard
) -> None:
    """Let daemon-hosted tools reuse the destination verified by their own TUI."""
    socket = client.socket_path()
    sessions = client.call("session", "list", "--json").get("sessions", [])
    matches = [
        row
        for row in sessions
        if row.get("running")
        and row.get("socket_path")
        and str(Path(row["socket_path"]).resolve()) == socket
    ]
    if len(matches) != 1 or not matches[0].get("name"):
        raise ValueError("cannot identify the inherited Herdr server for binding")
    record = {
        "version": 1,
        "herdr_session": matches[0]["name"],
        "pane_id": guard.pane["pane_id"],
        "terminal_id": guard.pane["terminal_id"],
        "processes": guard.stamp,
    }
    path = binding_path(env, identity)
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and previous != record:
        raise ValueError("native session is already bound to another destination")
    guard.check()
    guard.claim(allow_create=True)
    Store.write_json(path, record)


def _sync(
    env: Mapping[str, str],
    identity: str,
    client: Herdr,
    pane: dict[str, Any],
    guard: Guard,
    *,
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
        # Initial related/unassigned labels have our tokens but no parent owner.
        # Clearing them must not claim the parent or remove unrelated metadata.
        has_label = any(existing.get(key) for key in TOKEN_KEYS)
        if not has_label and not any(key in existing for key in INBOX_KEYS):
            return {"status": "unassigned", "pane": pane["pane_id"]}
        if registry(env) != snapshot:
            raise ValueError("parent registry changed before label cleanup; retry sync")
        pane = clear_inbox(client, pane, guard)
        if not has_label:
            return {"status": "updated", "pane": pane["pane_id"]}
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
    guard.claim(
        allow_create=confirm_parent,
        allow_move=session_start and identity.startswith("claude:"),
    )
    summary = parent_summary(env, key, row)
    name = board_name(env, repo, int(number))
    # Do not project an assignment that changed while reading its record.
    if registry(env) != snapshot:
        raise ValueError("parent registry changed during label preparation; retry sync")
    if session_start and previous_owner != token:
        pane = clear_inbox(client, pane, guard)
    result = client.publish(
        pane,
        agent=identity.split(":", 1)[0],
        display=f"{name} · {summary}" if name else summary,
        tokens={
            "tl_name": name or "",
            "tl_parent": key,
            "tl_issue": issue_label(int(number)),
            "tl_topic": summary,
            "tl_engine": identity.split(":", 1)[0],
            "tl_owner": token,
        },
        validate=guard.check,
    )
    return {
        **result,
        "parent": int(number),
        "summary": summary,
        "name": name,
        "name_status": "assigned" if name else "unassigned",
    }
