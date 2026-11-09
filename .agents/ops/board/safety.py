"""Public board data must not carry runtime identities or local paths."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .errors import BoardError

# Include the NFKC forms of fullwidth symbols: worker-log checks normalize text.
# A slash between Japanese words/symbols is prose. An ASCII path component such
# as /home still starts a path even immediately after Japanese text.
_FULLWIDTH_SYMBOLS = "".join(
    unicodedata.normalize("NFKC", chr(c))
    for c in range(0xFF01, 0xFF61)
    if not chr(c).isalnum() and c != 0xFF0F
)
_JAPANESE_LETTERS = (
    r"\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff"
    r"\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9f"
    r"\U0001b000-\U0001b16f\U00020000-\U0003ffff"
)
_JAPANESE_SEPARATOR = (
    "["
    + _JAPANESE_LETTERS
    + r"\u3000-\u303f\uff01-\uffef○×◎△"
    + re.escape(_FULLWIDTH_SYMBOLS)
    + "]"
)
ABSOLUTE_PATH = re.compile(
    r"(?<![a-zA-Z0-9_./])(?:"
    r"(?<!https:)(?<!http:)"
    rf"(?!(?<={_JAPANESE_SEPARATOR})/(?:[{_JAPANESE_LETTERS}]|"
    rf"(?![.~\\．～＼]){_JAPANESE_SEPARATOR}(?![a-zA-Z0-9_.~/\\])))/(?=\S)"
    r"|~[/\\]|[a-zA-Z]:[/\\]|\\\\)",
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
