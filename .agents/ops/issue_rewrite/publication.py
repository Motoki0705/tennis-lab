"""Allowlisted publication only; raw model output and diagnostics remain private."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|ya29\.[A-Za-z0-9_.-]{10,}|1//[A-Za-z0-9_-]{10,}"
    r"|1/[A-Za-z0-9_-]{20,}|GOCSPX-[A-Za-z0-9_-]{10,}"
    r"|AIza[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,})"
)
LOCAL_PATH = re.compile(
    r"(?<![\w:])(?:/(?:home|root|tmp|var|mnt|Users|private|opt|run)/[^\s\"'<>]+"
    r"|[A-Za-z]:\\[^\s\"'<>]+)"
)
SESSION = re.compile(
    r"(?i)\b(?:conversation[_ -]?id|session[_ -]?id)\b"
    r"(?:[\"'\s]*[:=][\"'\s]*[^\s,\"'<>]+)?"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


def publication_errors(text: str) -> list[str]:
    return [
        name
        for name, pattern in (
            ("token_like_output", TOKEN),
            ("local_path_output", LOCAL_PATH),
            ("session_identifier_output", SESSION),
        )
        if pattern.search(text)
    ]


def public_text(text: str) -> str:
    for pattern in (TOKEN, LOCAL_PATH, SESSION):
        text = pattern.sub("[redacted]", text)
    return text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as handle:
        path.chmod(0o600)
        json.dump(
            value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        handle.write("\n")


def diagnostic_differences(
    verdict: dict[str, Any], original: str, candidate: str
) -> list[dict[str, str]]:
    """Only publish excerpts that actually occur in the compared bodies.

    A verifier's free-text explanation can contain private CLI context, so keep
    it in the private run and publish the difference kind and matching excerpts.
    """
    return [
        {
            "kind": item["kind"],
            "original": public_text(item["original"])
            if item["original"] in original
            else "[non-excerpt omitted]",
            "rewritten": public_text(item["rewritten"])
            if item["rewritten"] in candidate
            else "[non-excerpt omitted]",
        }
        for item in verdict["differences"]
    ]


def summary(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    # JSON inside escaped HTML avoids rendering arbitrary issue/model Markdown.
    import html

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "## Issue rewrite\n\n<pre>"
            + html.escape(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
            )
            + "</pre>\n"
        )
