"""Public board data must not carry runtime identities or local paths."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote

from .errors import BoardError
from .prose import single_backtick_openers

MAX_SUBJECT_LENGTH = 200

# Only letters can precede a Japanese word separator. Punctuation, including
# fullwidth punctuation, must still introduce Japanese-named absolute paths.
_JAPANESE_LETTERS = (
    r"\u3005-\u3007\u303b\u3041-\u3096\u3099-\u309a\u309d-\u309f"
    r"\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9f"
    r"\U0001b000-\U0001b16f\U00020000-\U0003ffff"
)
ABSOLUTE_PATH = re.compile(
    r"(?<![a-zA-Z0-9_./])(?:"
    r"(?<!https:)(?<!http:)"
    rf"(?!(?<=[{_JAPANESE_LETTERS}])/[{_JAPANESE_LETTERS}])/(?=\S)"
    r"|~[/\\]|[a-zA-Z]:[/\\]|\\\\)",
    re.IGNORECASE,
)
_CODE_IDENTIFIER = re.compile(r"`[A-Za-z_][A-Za-z0-9_]*`")
_SENTENCE_PUNCTUATION = ".,:;!?．，：；！？"
_CODE_IDENTIFIER_LIST = re.compile(
    rf"(?<![^\s(（「『、。{_SENTENCE_PUNCTUATION}])"
    rf"{_CODE_IDENTIFIER.pattern}(?:[/／]{_CODE_IDENTIFIER.pattern})+"
    # Punctuation can end a sentence when followed by a real boundary.
    rf"(?=[{_SENTENCE_PUNCTUATION}]*(?:$|[\s)）」』、。]))"
)
SESSION_TOKEN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\b(?:session|thread)[_-]?id\s*[:=]",
    re.IGNORECASE,
)


def _path_view(value: str) -> str:
    if not _CODE_IDENTIFIER_LIST.search(value):
        return value
    openers = single_backtick_openers(value)

    def separators(match: re.Match[str]) -> str:
        if all(
            match.start() + identifier.start() in openers
            for identifier in _CODE_IDENTIFIER.finditer(match[0])
        ):
            return match[0].replace("/", " ").replace("／", " ")
        return match[0]

    return _CODE_IDENTIFIER_LIST.sub(separators, value)


def decoded_text(value: str) -> str:
    """Bound percent decoding consistently across board and worker-log checks."""
    for _ in range(4):
        new = unquote(value)
        if new == value:
            return value
        value = new
    if unquote(value) != value:
        raise ValueError("ambiguous_encoding")
    return value


def public(value: Any, parents: Mapping[str, Any], env: Mapping[str, str]) -> None:
    secrets = [row["session"].partition(":")[2] for row in parents.values()]
    secrets.extend(
        env.get(key, "")
        for key in (
            "CLAUDE_CODE_SESSION_ID",
            "CODEX_SESSION_ID",
            "CODEX_THREAD_ID",
            "ANTIGRAVITY_CONVERSATION_ID",
            "ANTIGRAVITY_TRAJECTORY_ID",
            "ANTIGRAVITY_CSRF_TOKEN",
        )
    )

    def check(item: Any) -> None:
        if isinstance(item, str):
            if any(
                secret and secret in item for secret in secrets
            ) or SESSION_TOKEN.search(item):
                raise BoardError("board text contains private session information")
            # A slash between standalone code identifiers is prose punctuation.
            # Recognize only literal syntax before decoding or normalization, so
            # encoded paths and slashes inside code still reach the path check.
            literal = _path_view(item)
            try:
                path_text = unicodedata.normalize("NFKC", decoded_text(literal))
            except ValueError:
                raise BoardError("board text contains ambiguous encoding") from None
            if any(
                ABSOLUTE_PATH.search(view) or view.strip().strip("\"'`") == "/"
                for view in (literal, path_text)
            ):
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


def truncate(value: str, limit: int) -> str:
    """Shorten already validated text without cutting an identifier list in half."""
    prefix = value[:limit]
    if len(prefix) < len(value):
        for match in _CODE_IDENTIFIER_LIST.finditer(value):
            if match.start() < len(prefix) < match.end():
                return value[: match.start()].rstrip()
    return prefix


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
        or len(value) > (MAX_SUBJECT_LENGTH if subject else 60000)
    ):
        raise BoardError("board text is empty or too long")
    if subject and ("\n" in value or "\t" in value):
        raise BoardError("board subject must be a single line")
    public(value, parents, env)
    return value
