"""Parent lifecycle integration and read-only public identity/roster views."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import unicodedata
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from herdr_labels.client import Herdr
from herdr_labels.coordination import owner_token, read_record, state_root
from herdr_labels.service import registry
from issues import CHILD_LABEL, PARENT_LABEL, parse_title
from lib.identity import session_identity
from parent.github import GitHub

from . import activity
from .errors import BoardError
from .names import Assignment, canonical_name, ensure_name, for_parent, retire_name
from .storage import Store, directory

STATES = {"idle", "working", "blocked", "done"}
REFERENCE = re.compile(r"([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#([1-9][0-9]*)")
ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'`(=：])(?:/|~[/\\]|[a-zA-Z]:[/\\]|\\\\)")


def _reference(key: str) -> tuple[str, int]:
    match = REFERENCE.fullmatch(key)
    if match is None:
        raise BoardError("invalid parent registry reference")
    return match[1].casefold(), int(match[2])


def _public(value: Any, parents: Mapping[str, Any], env: Mapping[str, str]) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    secrets = [row["session"].partition(":")[2] for row in parents.values()]
    secrets.extend(
        env.get(key, "")
        for key in ("CLAUDE_CODE_SESSION_ID", "CODEX_SESSION_ID", "CODEX_THREAD_ID")
    )
    if any(secret and secret in serialized for secret in secrets):
        raise BoardError("board output contains private session information")
    if ABSOLUTE_PATH.search(serialized):
        raise BoardError("board output contains a private path")


def _identifier(value: str, parents: Mapping[str, Any], env: Mapping[str, str]) -> str:
    if (
        not value
        or any(unicodedata.category(c).startswith("C") for c in value)
        or ABSOLUTE_PATH.search(value)
    ):
        raise BoardError("parent identifier must be public text without paths")
    _public(value, parents, env)
    return value


def sync_parent(
    env: Mapping[str, str], repo: str, parent: int, *, now: datetime | None = None
) -> dict[str, Any]:
    """Run after a successful parent action and after releasing its registry lock.

    Fetch GitHub metadata before taking the board write lock, then recheck the
    owner, status and generation so a delayed callback cannot update a new owner
    or resurrect a name after release/ended.
    """
    identity = session_identity(env)
    key = f"{repo.casefold()}#{parent}"
    parents = registry(env)
    expected = parents.get(key)
    if expected is None or expected["session"] != identity:
        raise BoardError("parent assignment changed before board synchronization")
    identifier = None
    if expected["status"] == "active":
        issue = GitHub(repo).issue(parent)
        labels = {label["name"] for label in issue["labels"]}
        if labels & {PARENT_LABEL, CHILD_LABEL}:
            identifier = _identifier(
                parse_title(issue["title"]).identifier or "", parents, env
            )
        # Standalone issues have no own identifier, even if their arbitrary
        # title happens to look like the parent/child naming convention.
    store = Store(directory(env))
    with store.transaction() as connection:
        parents = registry(env)
        row = parents.get(key)
        if row is None or any(
            row[field] != expected[field]
            for field in ("session", "status", "generation")
        ):
            raise BoardError("parent assignment changed before board synchronization")
        clock = now if now is not None else datetime.now(UTC)
        previous = for_parent(connection, repo, parent)
        assignment: Assignment | None
        if row["status"] == "active":
            if identifier is not None:
                _identifier(identifier, parents, env)
            assignment = ensure_name(connection, repo, parent, identifier, now=clock)
            status = (
                "retained" if previous and previous.id == assignment.id else "assigned"
            )
        elif row["status"] == "ended":
            assignment = retire_name(connection, repo, parent, now=clock)
            status = "retired" if assignment else "unassigned"
        else:
            assignment, status = previous, "retained" if previous else "unassigned"
        result = {
            "status": status,
            "name": assignment.name if assignment else None,
            "repo": repo.casefold(),
            "parent": parent,
        }
        _public(result, parents, env)
        return result


def _assigned(connection: sqlite3.Connection, repo: str, parent: int) -> Assignment:
    assignment = for_parent(connection, repo, parent)
    if assignment is None or assignment.retired_at is not None:
        raise BoardError("active parent has no live board name; retry parent register")
    canonical_name(assignment.name)
    return assignment


def whoami(env: Mapping[str, str]) -> dict[str, str | int]:
    identity = session_identity(env)
    parents = registry(env)
    owned = [
        key
        for key, row in parents.items()
        if row["session"] == identity and row["status"] == "active"
    ]
    if len(owned) != 1:
        raise BoardError(
            "only the registered active parent session has a board identity"
        )
    repo, parent = _reference(owned[0])
    with Store(directory(env)).read() as connection:
        assignment = _assigned(connection, repo, parent)
        result: dict[str, str | int] = {
            "name": assignment.name,
            "repo": repo,
            "parent": parent,
        }
    if registry(env) != parents:
        raise BoardError("parent registry changed during board lookup; retry")
    _public(result, parents, env)
    return result


def _runtime_state(
    root: Path, scope: str, identity: str, owner: dict[str, Any], agents: list[Any]
) -> str:
    candidates = [
        agent
        for agent in agents
        if isinstance(agent, dict)
        and agent.get("terminal_id") == owner["terminal_id"]
        and agent.get("agent") == identity.partition(":")[0]
    ]
    if len(candidates) != 1:
        return "unknown"
    agent = candidates[0]
    token = agent.get("tokens", {}).get("tl_owner")
    expected = owner_token(identity)
    runtime_key = owner_token(scope + ":" + owner["terminal_id"])
    runtime = read_record(root / "panes" / f"{runtime_key}.json")
    if runtime is not None and (
        runtime.get("owner") != expected or runtime.get("state") != "active"
    ):
        return "unknown"
    if token != expected and (token or runtime is None):
        return "unknown"
    # Native references, when available, fence a replaced conversation even if
    # its terminal still has old display metadata. Never expose the reference.
    native = agent.get("agent_session")
    if native is not None and (
        not isinstance(native, dict)
        or native.get("kind") != "id"
        or native.get("agent") != identity.partition(":")[0]
        or native.get("value") != identity.partition(":")[2]
    ):
        return "unknown"
    state = agent.get("state")
    return state if isinstance(state, str) and state in STATES else "unknown"


def _runtime_states(env: Mapping[str, str], identities: list[str]) -> dict[str, str]:
    """Join confirmed owners to server scope and terminal, never pane focus/title.

    Runtime data is advisory. Missing/malformed/unavailable Herdr data has the
    single specified representation, unknown, and never changes the registry.
    """
    result = dict.fromkeys(identities, "unknown")
    records: dict[str, dict[str, Any]] = {}
    root = state_root(env) / "herdr-labels"
    for identity in identities:
        try:
            record = read_record(root / "owners" / f"{owner_token(identity)}.json")
            if (
                record
                and isinstance(record.get("scope"), str)
                and isinstance(record.get("terminal_id"), str)
            ):
                records[identity] = record
        except (OSError, ValueError, TypeError):
            continue
    if not records:
        return result
    try:
        sessions = Herdr(env).call("session", "list", "--json").get("sessions", [])
        if not isinstance(sessions, list):
            return result
        for server in sessions:
            if (
                not isinstance(server, dict)
                or not server.get("running")
                or not isinstance(server.get("socket_path"), str)
                or not isinstance(server.get("name"), str)
            ):
                continue
            scope = owner_token(str(Path(server["socket_path"]).resolve()))
            matching = {
                identity: row
                for identity, row in records.items()
                if row["scope"] == scope
            }
            if not matching:
                continue
            try:
                agents = Herdr(env, server["name"]).call("agent", "list").get("agents")
                if not isinstance(agents, list):
                    continue
                for identity, owner in matching.items():
                    try:
                        result[identity] = _runtime_state(
                            root, scope, identity, owner, agents
                        )
                    except (OSError, ValueError, KeyError, TypeError, AttributeError):
                        continue
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                subprocess.SubprocessError,
            ):
                continue
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return result
    return result


def agents(env: Mapping[str, str]) -> dict[str, list[dict[str, Any]]]:
    parents = registry(env)
    active = [(key, row) for key, row in parents.items() if row["status"] == "active"]
    states = _runtime_states(env, [row["session"] for _, row in active])
    entries = []
    store = Store(directory(env))
    snapshot = (
        store.read()
        if store.path.exists() or store.path.is_symlink()
        else nullcontext(None)
    )
    with snapshot as connection:
        for key, row in sorted(active):
            repo, parent = _reference(key)
            engine, separator, session = row["session"].partition(":")
            if engine not in {"claude", "codex"} or not separator or not session:
                raise BoardError("invalid parent engine identity")
            assignment = (
                for_parent(connection, repo, parent) if connection is not None else None
            )
            if assignment is not None and assignment.retired_at is not None:
                assignment = None
            if assignment is not None:
                canonical_name(assignment.name)
            state = (
                activity.summary(connection, assignment.id)
                if connection is not None and assignment is not None
                else activity.Activity()
            )
            entries.append(
                {
                    "name": assignment.name if assignment else None,
                    "name_status": "assigned" if assignment else "unassigned",
                    "repo": repo,
                    "parent": parent,
                    "identifier": _identifier(assignment.identifier, parents, env)
                    if assignment and assignment.identifier is not None
                    else None,
                    "engine": engine,
                    "state": states[row["session"]],
                    "unread_count": state.unread_count,
                    "last_activity": state.last_post_at,
                }
            )
    if registry(env) != parents:
        raise BoardError("parent registry changed during roster lookup; retry")
    result = {"agents": entries}
    _public(result, parents, env)
    return result
