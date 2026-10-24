"""Public board data must not carry runtime identities or local paths."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .errors import BoardError

ABSOLUTE_PATH = re.compile(
    r"(?<![a-zA-Z0-9_./])(?:(?<!https:)(?<!http:)/(?=\S)|~[/\\]|[a-zA-Z]:[/\\]|\\\\)",
    re.IGNORECASE,
)
SESSION_TOKEN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\b(?:session|thread)[_-]?id\s*[:=]",
    re.IGNORECASE,
)


def public(value: Any, parents: Mapping[str, Any], env: Mapping[str, str]) -> None:
    secrets = [row["session"].partition(":")[2] for row in parents.values()]
    secrets.extend(
        env.get(key, "")
        for key in ("CLAUDE_CODE_SESSION_ID", "CODEX_SESSION_ID", "CODEX_THREAD_ID")
    )

    def check(item: Any) -> None:
        if isinstance(item, str):
            if any(
                secret and secret in item for secret in secrets
            ) or SESSION_TOKEN.search(item):
                raise BoardError("board text contains private session information")
            if ABSOLUTE_PATH.search(item) or item.strip().strip("\"'`") == "/":
                raise BoardError("board text contains a private path")
            if any(
                unicodedata.category(c).startswith("C") and c not in "\n\t"
                for c in item
            ):
                raise BoardError("board text contains invalid control characters")
        elif isinstance(item, Mapping):
            for key, entry in item.items():
                check(key)
                check(entry)
        elif isinstance(item, (list, tuple)):
            for entry in item:
                check(entry)

    check(value)


def text(
    value: str,
    parents: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    subject: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > (200 if subject else 60000)
    ):
        raise BoardError("board text is empty or too long")
    if subject and ("\n" in value or "\t" in value):
        raise BoardError("board subject must be a single line")
    public(value, parents, env)
    return value
