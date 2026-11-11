"""Public projections and privacy checks; never read live GitHub or launch agents."""

from __future__ import annotations

import copy
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from board.safety import SESSION_TOKEN
from worker_logs.journal import PublicationError
from worker_logs.safety import URL, Guard, decoded

WORKTREE = re.compile(r"\.claude[/\\]worktrees[/\\][^\s`\"';,。)\]}]+")
PRIVATE_ROOT = re.compile(
    r"(?:/(?:home|Users|tmp|var/tmp)/|[A-Za-z]:[/\\]Users[/\\])[^\s\"'`]+"
)
UUID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)


def text_view(value: str) -> str:
    return unicodedata.normalize("NFKC", decoded(value))


def validation_view(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: validation_view(v) for k, v in value.items()}
    if isinstance(value, list):
        return [validation_view(v) for v in value]
    if not isinstance(value, str):
        return value

    def example(match: re.Match[str]) -> str:
        # Reserved example URLs in gold/reproductions describe a defect, not a
        # credential for a real service. No real host/userinfo is exempted.
        url = urlsplit(match[0])
        if (
            url.hostname == "example.com"
            and url.username == "user"
            and url.password == "pass"
        ):
            return "[documented credential example]"
        return match[0]

    return URL.sub(example, value).replace("/.agents/", "repo-root:.agents/")


def check_public(value: Any, *, source_text: bool = False) -> None:
    def inspect(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                inspect(key)
                inspect(item)
        elif isinstance(node, list):
            for item in node:
                inspect(item)
        elif isinstance(node, str):
            view = text_view(node)
            worktrees = list(WORKTREE.finditer(view))
            if any(
                not source_text
                or not any(marker in m[0] for marker in ("<", "$", "{", "…", "..."))
                for m in worktrees
            ):
                raise ValueError("public evaluation material contains a worktree path")
            if source_text and (PRIVATE_ROOT.search(view) or UUID.search(view)):
                raise ValueError(
                    "public source/document contains a private path or identity"
                )

    inspect(value)
    identities = [
        os.environ.get(k, "")
        for k in ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDE_CODE_SESSION_ID")
    ]
    try:
        # Captured JSON is data: validate every value strictly. Code/prose may
        # contain system executables, schema field names and operator examples;
        # concrete local roots, worktrees and native UUIDs are still forbidden.
        if not source_text:
            Guard(private=identities).check(validation_view(value))
        elif any(identity and identity in str(value) for identity in identities):
            raise ValueError("public source/document contains a native identity")
    except PublicationError as exc:
        raise ValueError("unsafe public evaluation material: " + exc.code) from None


def project_input(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    counts = {"worktree_paths": 0, "native_identifiers": 0, "local_paths": 0}

    def project(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: project(item) for key, item in node.items()}
        if isinstance(node, list):
            return [project(item) for item in node]
        if not isinstance(node, str) or not node:
            return copy.deepcopy(node)
        clean, count = WORKTREE.subn("[worktree path omitted]", node)
        counts["worktree_paths"] += count
        clean, count = SESSION_TOKEN.subn("[native identity omitted]", clean)
        counts["native_identifiers"] += count
        sanitized = Guard().sanitize(clean)
        counts["local_paths"] += sum(sanitized.local_paths.values())
        return sanitized.text

    public = project(value)
    check_public(public)
    return public, counts


def check_files(repo: Path, paths: list[str]) -> dict[str, Any]:
    from lib.review_tools.retrieval import strict_json

    checked = []
    for name in sorted(set(paths)):
        path = repo / name
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(repo.resolve()):
            raise ValueError("public file must be a regular checkout file")
        try:
            if path.suffix == ".json":
                check_public(strict_json(path.read_bytes()))
            else:
                check_public(path.read_text(encoding="utf-8"), source_text=True)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"public check failed for {name}: {exc}") from None
        checked.append(name)
    return {"schema_version": 1, "checked_files": checked, "agent_invocations": 0}
