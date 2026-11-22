"""Reuse worker-log sanitization for the entire public record and metadata."""

from __future__ import annotations

from collections import Counter
from typing import Any

from worker_logs.journal import require
from worker_logs.safety import Guard
from worker_logs.schema import fields

from .schema import validate_record


def validate_sanitization(value: Any) -> None:
    fields(value, {"version", "replacements", "local_paths"})
    require(
        type(value["version"]) is int and value["version"] == 2, "record_sanitization"
    )
    require(
        type(value["replacements"]) is int and value["replacements"] >= 0,
        "record_sanitization",
    )
    paths = value["local_paths"]
    require(
        isinstance(paths, dict)
        and set(paths) <= {"posix", "home", "windows", "unc"}
        and all(type(n) is int and n > 0 for n in paths.values())
        and sum(paths.values()) <= value["replacements"],
        "record_sanitization",
    )


def validate_public(value: Any, guard: Guard | None = None) -> dict[str, Any]:
    fields(value, {"record", "sanitization"})
    record = validate_record(value["record"])
    validate_sanitization(value["sanitization"])
    (guard or Guard()).check(value)
    return {"record": record, "sanitization": dict(value["sanitization"])}


def render(record: dict[str, Any], guard: Guard | None = None) -> dict[str, Any]:
    """Preserve all fields; unsafe transformations fail instead of truncating."""
    original = validate_record(record)
    checker = guard or Guard()
    replacements = 0
    paths: Counter[str] = Counter()

    prose = {
        "summary",
        "reason",
        "target",
        "instruction",
        "conditions",
        "expected_result",
        "reuse_scope",
        "recheck_scope",
        "unresolved",
        "resume_conditions",
    }

    def sanitize(value: Any, free_text: bool = False) -> Any:
        nonlocal replacements
        if isinstance(value, dict):
            # Keys are contract identifiers; never silently rename them.
            checker.check(list(value))
            return {k: sanitize(v, k in prose) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v, free_text) for v in value]
        if isinstance(value, str) and free_text:
            result = checker.sanitize(value)
            replacements += result.replacements
            paths.update(result.local_paths)
            return result.text
        checker.check(value)
        return value

    projected = {
        "record": sanitize(original),
        "sanitization": {
            "version": 2,
            "replacements": replacements,
            "local_paths": dict(paths),
        },
    }
    return validate_public(projected, checker)
