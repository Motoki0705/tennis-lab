"""Shared, dependency-light value checks used by knowledge writers and readers."""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any
from urllib.parse import urlsplit

TASK_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ID_RE = re.compile(r"^(?:run|group)-[a-z0-9]+(?:-[a-z0-9]+)*$")
NODE_TYPES = {"run", "group"}
STATUSES = {"done", "failed", "running", "planned"}
PROVIDERS = {"claude", "codex", "gemini", "human", "other"}


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def iso_date(value: Any) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        return False
    try:
        date.fromisoformat(str(value))
    except ValueError:
        return False
    return True


def http_url(value: Any) -> bool:
    if not nonempty_text(value):
        return False
    try:
        url = urlsplit(value)
        return url.scheme in {"http", "https"} and bool(url.hostname)
    except ValueError:
        return False


def json_value(value: Any) -> bool:
    """Match values that can be passed through the Web UI's JSON boundary."""
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(json_value(item) for item in value)
    if isinstance(value, dict):
        return all(nonempty_text(k) and json_value(v) for k, v in value.items())
    return False


def has_prose(body: str) -> bool:
    """Reject empty scaffolds, without dictating the wording or heading order."""
    visible = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    return any(line.strip() and not line.lstrip().startswith("#") for line in visible.splitlines())
