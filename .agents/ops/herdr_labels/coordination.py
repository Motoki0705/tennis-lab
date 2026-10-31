"""Private display destinations and serialized pane lifecycle events."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from parent.state import Store

from .client import Herdr


class LifecycleCallerError(ValueError):
    """A lifecycle event cannot be attributed to the pane's foreground agent."""


def state_root(env: Mapping[str, str]) -> Path:
    base = Path(env.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents"


def owner_token(identity: str) -> str:
    return hashlib.sha256(identity.encode()).hexdigest()


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as stream:
        os.chmod(path, 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def process_stamp(client: Herdr, pane: dict[str, Any]) -> list[dict[str, Any]]:
    info = client.call("pane", "process-info", "--pane", pane["pane_id"])[
        "process_info"
    ]
    stamps = []
    for proc in info["foreground_processes"]:
        if proc["name"] != pane.get("agent"):
            continue
        pid = int(proc["pid"])
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        stamps.append({"pid": pid, "start": stat.rsplit(")", 1)[1].split()[19]})
    if len(stamps) != 1:
        raise ValueError("cannot identify exactly one foreground agent process")
    return stamps


def read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("invalid display coordination record")
    return value


class Guard:
    def __init__(
        self, env: Mapping[str, str], identity: str, client: Herdr, pane: dict[str, Any]
    ) -> None:
        self.root = state_root(env) / "herdr-labels"
        self.identity = identity
        self.owner = owner_token(identity)
        self.native_verified = False
        self.client, self.pane = client, pane
        self.scope = owner_token(client.socket_path())
        self.stamp = process_stamp(client, pane)
        self.destination = {
            "version": 1,
            "scope": self.scope,
            "terminal_id": pane["terminal_id"],
        }
        key = owner_token(self.scope + ":" + pane["terminal_id"])
        self.path = self.root / "panes" / (key + ".json")
        self.runtime: dict[str, Any] = {}

    def claim(self, *, allow_create: bool = False, allow_move: bool = False) -> None:
        """Keep the confirmed parent pane even after display clear/SessionEnd."""
        path = self.root / "owners" / (self.owner + ".json")
        previous = read_record(path)
        if allow_move and previous is not None and previous != self.destination:
            if (
                not self.native_verified
                or self.client.env.get("HERDR_ENV") != "1"
                or not self.identity.startswith("claude:")
            ):
                raise ValueError("display moves require a verified Claude SessionStart")
            self._require_vacated(previous)
            with locked(path.with_suffix(".lock")):
                if read_record(path) != previous:
                    raise ValueError(
                        "confirmed parent display changed during move; retry"
                    )
                self.check()
                Store.write_json(path, self.destination)
            return
        with locked(path.with_suffix(".lock")):
            previous = read_record(path)
            if previous is not None and previous != self.destination:
                raise ValueError(
                    "confirmed parent display belongs to another pane; "
                    "inspect the assignment before explicitly rebinding"
                )
            if previous is None:
                if not allow_create:
                    raise ValueError(
                        "parent display destination is unconfirmed; "
                        "register the parent or explicitly bind the inspected pane"
                    )
                Store.write_json(path, self.destination)

    def _require_vacated(self, previous: dict[str, Any]) -> None:
        """Absence must be verified against the recorded server, never inferred."""
        native = self.pane.get("agent_session")
        if native is not None and (
            not isinstance(native, dict)
            or native.get("kind") != "id"
            or native.get("agent") != "claude"
            or native.get("value") != self.identity.partition(":")[2]
        ):
            raise ValueError("new pane native session differs from the caller")
        scope, terminal = previous.get("scope"), previous.get("terminal_id")
        if (
            not isinstance(scope, str)
            or re.fullmatch(r"[0-9a-f]{64}", scope) is None
            or not isinstance(terminal, str)
            or not terminal
        ):
            raise ValueError("invalid confirmed parent display destination")
        servers = self.client.call("session", "list", "--json").get("sessions")
        if not isinstance(servers, list):
            raise ValueError("invalid Herdr server list; cannot verify old destination")
        matches = []
        for server in servers:
            if not isinstance(server, dict) or type(server.get("running")) is not bool:
                raise ValueError(
                    "invalid Herdr server entry; cannot verify old destination"
                )
            if not server["running"]:
                continue
            socket, name = server.get("socket_path"), server.get("name")
            if (
                not isinstance(socket, str)
                or not socket
                or not Path(socket).is_absolute()
                or not isinstance(name, str)
                or not name
            ):
                raise ValueError(
                    "invalid Herdr server identity; cannot verify old destination"
                )
            if owner_token(str(Path(socket).resolve())) == scope:
                matches.append(name)
        if len(matches) > 1:
            raise ValueError("ambiguous Herdr server for the old destination")
        if not matches:
            return  # A validated server list confirms the old server is stopped.
        client = Herdr(self.client.env, matches[0])
        agents = client.call("agent", "list").get("agents")
        if not isinstance(agents, list) or not all(
            isinstance(agent, dict)
            and all(
                isinstance(agent.get(key), str) and agent[key]
                for key in ("pane_id", "terminal_id", "agent")
            )
            and isinstance(agent.get("tokens", {}), dict)
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in agent.get("tokens", {}).items()
            )
            for agent in agents
        ):
            raise ValueError("invalid Herdr agent list; cannot verify old destination")
        candidates = [agent for agent in agents if agent["terminal_id"] == terminal]
        if len(candidates) > 1:
            raise ValueError("ambiguous Herdr terminal for the old destination")
        if not candidates:
            return
        pane = candidates[0]
        live = pane.get("tokens", {}).get("tl_owner") == self.owner
        key = owner_token(scope + ":" + terminal)
        runtime = read_record(self.root / "panes" / (key + ".json"))
        if (
            not live
            and runtime is not None
            and runtime.get("owner") == self.owner
            and runtime.get("state") == "active"
            and pane["agent"] == "claude"
        ):
            stamps = runtime.get("processes")
            if (
                not isinstance(stamps, list)
                or len(stamps) != 1
                or not isinstance(stamps[0], dict)
                or type(stamps[0].get("pid")) is not int
                or not isinstance(stamps[0].get("start"), str)
                or not stamps[0]["start"]
            ):
                raise ValueError(
                    "invalid old pane process stamp; cannot verify destination"
                )
            live = process_stamp(client, pane) == runtime["processes"]
        if live:
            raise ValueError(
                "the same session still occupies another live pane; "
                "inspect duplicated sessions before synchronizing"
            )

    def check(self) -> None:
        if process_stamp(self.client, self.pane) != self.stamp:
            raise ValueError("foreground agent changed during display synchronization")
        if read_record(self.path) != self.runtime:
            raise ValueError("pane session changed during display synchronization")

    def require_native_ancestor(self) -> None:
        """A lifecycle payload must originate from the pane's current agent."""
        try:
            native = self.stamp[0]["pid"]
            pid = os.getpid()
            while pid > 1:
                if pid == native:
                    self.native_verified = True
                    return
                stat = (Path("/proc") / str(pid) / "stat").read_text()
                command = stat.split("(", 1)[1].rsplit(")", 1)[0]
                if command == self.pane.get("agent"):
                    raise ValueError(
                        "nested agent cannot report lifecycle for its ancestor's pane"
                    )
                pid = int(stat.rsplit(")", 1)[1].split()[1])
            raise ValueError("lifecycle hook is not a descendant of this pane's agent")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise LifecycleCallerError(str(exc)) from exc

    @contextmanager
    def hold(
        self, *, start: bool = False, end: bool = False, confirmed: bool = False
    ) -> Iterator[Guard]:
        # Every sync/clear/start/end uses this lock, including title lookup.
        # A newer SessionStart either follows the old write or invalidates it
        # before it can run; SessionEnd also fences late old-session commands.
        with locked(self.path.with_suffix(".lock")):
            if process_stamp(self.client, self.pane) != self.stamp:
                raise ValueError(
                    "foreground agent changed while awaiting the pane lock"
                )
            if confirmed:
                self.claim(allow_create=True)
            fresh = self.client.pane(self.pane["pane_id"])
            if (fresh["terminal_id"], fresh.get("agent")) != (
                self.pane["terminal_id"],
                self.pane.get("agent"),
            ):
                raise ValueError("pane occupant changed while awaiting the pane lock")
            self.pane = fresh
            previous = read_record(self.path)
            expected = {
                "version": 1,
                "owner": self.owner,
                "processes": self.stamp,
                "state": "active",
            }
            if start or previous is None:
                Store.write_json(self.path, expected)
            elif previous != expected:
                raise ValueError("caller is no longer the active session in this pane")
            self.runtime = expected
            yield self
            self.check()
            if end:
                Store.write_json(self.path, {**expected, "state": "ended"})
