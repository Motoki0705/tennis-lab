"""Machine-local ownership: one locked registry, atomic durable writes."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .policy import object_value


def session_identity(env: dict[str, str]) -> str:
    identities = [
        (kind, env.get(name, "").strip())
        for kind, name in (
            ("claude", "CLAUDE_CODE_SESSION_ID"),
            ("codex", "CODEX_SESSION_ID"),
        )
        if env.get(name, "").strip()
    ]
    if len(identities) != 1:
        raise ValueError(
            "exactly one of CLAUDE_CODE_SESSION_ID / CODEX_SESSION_ID must identify this session; missing/ambiguous identity (do not ask the human for an ID)"
        )
    kind, value = identities[0]
    if kind == "codex" and env.get("CODEX_THREAD_ID", value).strip() != value:
        raise ValueError(
            "run parent operations from the Codex root session, not a subagent thread"
        )
    if any(ord(c) < 33 for c in value):
        raise ValueError("invalid session environment identity")
    return f"{kind}:{value}"


def state_directory() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents/parent-sessions"


class Store:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "registry.json"
        self.data: dict[str, Any] = {}

    @contextmanager
    def locked(self) -> Iterator[Store]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / "registry.lock").open("a", encoding="utf-8") as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = (
                object_value(json.loads(self.path.read_text(encoding="utf-8")))
                if self.path.exists()
                else {"version": 1, "parents": {}}
            )
            if set(self.data) != {"version", "parents"} or self.data["version"] != 1:
                raise ValueError("unknown or malformed parent-session state")
            parents: dict[str, Any] = {}
            for key, row in object_value(self.data["parents"]).items():
                canonical = key.casefold()
                if canonical in parents:
                    raise ValueError(
                        f"registry contains a repository alias collision: {canonical}"
                    )
                parents[canonical] = row
                if (
                    not isinstance(row, dict)
                    or not {"session", "status", "generation", "tasks", "reviews"}
                    <= row.keys()
                    or row["status"]
                    not in {"active", "released", "ended", "handoff_pending"}
                ):
                    raise ValueError(f"invalid parent-session record: {key}")
                if (
                    not isinstance(row["session"], str)
                    or not row["session"]
                    or not isinstance(row["tasks"], dict)
                    or not isinstance(row["reviews"], list)
                ):
                    raise ValueError(f"invalid parent-session data types: {key}")
            owners = [
                r["session"]
                for r in parents.values()
                if r["status"] in {"active", "handoff_pending"}
            ]
            if len(owners) != len(set(owners)):
                raise ValueError(
                    "registry contains multiple active parents for one session"
                )
            self.data["parents"] = parents
            yield self

    @property
    def parents(self) -> dict[str, Any]:
        return object_value(self.data["parents"])

    def save(self) -> None:
        self.write_json(self.path, self.data)

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(path.name + ".tmp")
        # State is private; never follow a pre-existing temporary symlink.
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
