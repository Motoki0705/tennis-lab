"""Schema for the four shared memory layers."""

from __future__ import annotations

import re
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import date
from html import unescape
from ipaddress import ip_address
from urllib.parse import unquote, urlsplit

import yaml

from .layout import (
    DEDICATED_AREA as DEDICATED_AREA,
)
from .layout import (
    INDEX_NAME as INDEX_NAME,
)
from .layout import (
    MEMORY_DIR as MEMORY_DIR,
)
from .layout import (
    SHARED_LAYERS as SHARED_LAYERS,
)
from .layout import (
    SUMMARY_PATH as SUMMARY_PATH,
)
from .layout import (
    is_entry_filename as is_entry_filename,
)
from .layout import (
    is_shared_path as is_shared_path,
)

LAYER_DESCRIPTIONS = {
    "workflow": "作業のコツ",
    "decision": "ユーザーの判断",
    "environment": "環境の癖",
    "gotcha": "落とし穴",
}
APPLIES_TO = frozenset({"all", "claude", "codex", "gemini"})
SUMMARY_MAX_LINES = 80
SUMMARY_MAX_BYTES = 8192
INDEX_MAX_ENTRIES = 30
ENTRY_MAX_LINES = 40
SUMMARY_ITEM_MAX_CHARS = 160
STALE_DAYS = 90

ENTRY_FIELDS = (
    "title",
    "summary",
    "applies_to",
    "source",
    "created",
    "last_verified",
    "evidence",
    "citations",
)
REQUIRED_KEYS = frozenset(ENTRY_FIELDS)
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)^---[ \t]*\r?$", re.MULTILINE | re.DOTALL)
_URL_AUTHORITY = re.compile(
    r"(?<![\w:/])(?:[a-z][a-z0-9+.-]*:)?//[^/\s<>\"'`?#()]+", re.IGNORECASE
)
_PRIVATE_DATA = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "token or API key": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{20,})"
    ),
    "credential assignment": re.compile(
        r"""\b(?:token|password|secret|api[_-]?key)["']?[ \t]*[:=][ \t]*\S""",
        re.IGNORECASE,
    ),
    "private key": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "machine-specific path": re.compile(
        r"\bfile:[/\\]|(?<![\w./-])/(?:home|root|Users|mnt/[A-Za-z]/Users)/"
        r"|\b[A-Za-z]:[\\/]Users[\\/]",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Citation:
    path: str
    symbol: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Entry:
    layer: str
    filename: str
    title: str
    summary: str
    applies_to: str
    source: str
    created: date
    last_verified: date
    evidence: str
    citations: tuple[Citation, ...]

    @property
    def path(self) -> str:
        return f"{MEMORY_DIR}/{self.layer}/{self.filename}"


class CitationError(ValueError):
    """Invalid structure or field in a citation."""


def _has_local_url(text: str) -> bool:
    for match in _URL_AUTHORITY.finditer(text):
        try:
            host = (urlsplit(match[0]).hostname or "").rstrip(".")
        except ValueError:
            continue
        if host in {"localhost", "localhost.localdomain"} or host.endswith(
            (".localhost", ".local", ".internal")
        ):
            return True
        try:
            address = ip_address(host)
        except ValueError:
            continue
        if not address.is_global:
            return True
    return False


def private_data_kinds(text: str) -> tuple[str, ...]:
    """Return categories only so diagnostics never copy the matched values."""
    decoded = unquote(unescape(text))
    views = (text,) if decoded == text else (text, decoded)
    kinds = [
        kind
        for kind, pattern in _PRIVATE_DATA.items()
        if any(pattern.search(view) for view in views)
    ]
    if any(_has_local_url(view) for view in views):
        kinds.append("local URL")
    return tuple(kinds)


def iso_date(value: object, field: str) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD)")


class _UniqueKeysLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: yaml.Node, deep: bool = False
    ) -> dict[Hashable, object]:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("expected a YAML mapping")
        mapping: dict[Hashable, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in mapping:
                raise ValueError("YAML mapping keys must be unique strings")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _string(value: object, field: str, *, one_line: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if one_line and value.splitlines() != [value]:
        raise ValueError(f"{field} must be on one line")
    if re.search(r"[\ud800-\udfff]", value):
        raise ValueError(f"{field} must contain valid Unicode")
    if kinds := private_data_kinds(value):
        raise ValueError(f"{field} contains {', '.join(kinds)}")
    return value


def _citation(value: object, field: str) -> Citation:
    if (
        not isinstance(value, dict)
        or "path" not in value
        or value.keys() - {"path", "symbol", "note"}
    ):
        raise ValueError(
            f"{field} requires path and accepts only path, symbol and note"
        )
    path = _string(value["path"], f"{field}.path", one_line=True)
    if (
        path.startswith("/")
        or "\\" in path
        or ".." in path.split("/")
        or "\0" in path
        or re.match(r"^[A-Za-z]:", path)
    ):
        raise ValueError(f"{field}.path must be a repository-relative path")
    return Citation(
        path,
        _string(value["symbol"], f"{field}.symbol") if "symbol" in value else None,
        _string(value["note"], f"{field}.note") if "note" in value else None,
    )


def read_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Read the YAML mapping and preserve the Markdown after its delimiter."""
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError("missing YAML frontmatter")
    try:
        data = yaml.load(match[1], Loader=_UniqueKeysLoader)
    except yaml.YAMLError:
        raise ValueError("invalid YAML frontmatter") from None
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a mapping")
    return data, text[match.end() :]


def parse_entry(layer: str, filename: str, text: str, *, today: date) -> Entry:
    """Parse frontmatter or raise ValueError with the field that needs repair."""
    if layer not in SHARED_LAYERS:
        raise ValueError("unknown shared memory layer")
    if not is_entry_filename(filename):
        raise ValueError("entry filename must be kebab-case.md")
    data, _ = read_frontmatter(text)
    if "citations" not in data:
        raise CitationError("missing frontmatter fields: citations")
    if not isinstance(data["citations"], list):
        raise CitationError("citations must be a list")
    try:
        citations = tuple(
            _citation(citation, f"citations[{index}]")
            for index, citation in enumerate(data["citations"])
        )
    except ValueError as exc:
        raise CitationError(str(exc)) from None
    if missing := REQUIRED_KEYS - data.keys():
        raise ValueError(f"missing frontmatter fields: {', '.join(sorted(missing))}")
    if "type" in data:
        raise ValueError("type is supplied by the layer directory")
    title = _string(data["title"], "title", one_line=True)
    summary = _string(data["summary"], "summary", one_line=True)
    if len(summary) > SUMMARY_ITEM_MAX_CHARS:
        raise ValueError(f"summary exceeds {SUMMARY_ITEM_MAX_CHARS} characters")
    scope = _string(data["applies_to"], "applies_to")
    if scope not in APPLIES_TO:
        raise ValueError(f"applies_to must be one of {', '.join(sorted(APPLIES_TO))}")
    created = iso_date(data["created"], "created")
    verified = iso_date(data["last_verified"], "last_verified")
    if not created <= verified <= today:
        raise ValueError("need created <= last_verified <= today")
    return Entry(
        layer=layer,
        filename=filename,
        title=title,
        summary=summary,
        applies_to=scope,
        source=_string(data["source"], "source"),
        created=created,
        last_verified=verified,
        evidence=_string(data["evidence"], "evidence"),
        citations=citations,
    )


def markdown_body(text: str) -> str:
    match = _FRONTMATTER.match(text)
    return text[match.end() :] if match else text
