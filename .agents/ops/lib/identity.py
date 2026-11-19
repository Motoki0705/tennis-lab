"""Environment-owned root session identity, shared by parent and board."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def native_runtime() -> dict[str, Any] | None:
    """Pin the native agent for a later forced local worktree stop.

    This is process provenance, not a substitute for session_identity. A CLI
    without a native ancestor (e.g. an API caller) has no manageable process.
    """
    import os

    from lib.opsproc import process_identity

    pid = os.getppid()
    while pid > 1:
        try:
            entry = Path(f"/proc/{pid}")
            raw = (entry / "stat").read_text()
            fields = raw[raw.rindex(")") + 2 :].split()
            name = raw[raw.index("(") + 1 : raw.rindex(")")]
            if name in {"codex", "claude"}:
                ident = process_identity(pid)
                cwd = os.readlink(entry / "cwd")
                if ident is not None:
                    return {"cwd": cwd, "process": ident}
            pid = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            break
    return None


def session_identity(env: Mapping[str, str]) -> str:
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
    if kind == "claude" and any(
        env.get(key, "").strip()
        for key in ("CLAUDE_CODE_PARENT_SESSION_ID", "CLAUDE_CODE_AGENT_ID")
    ):
        raise ValueError(
            "run parent operations from the Claude root session, not a nested agent"
        )
    if any(ord(c) < 33 for c in value):
        raise ValueError("invalid session environment identity")
    return f"{kind}:{value}"
