"""Subprocess helpers: every external command fails loudly with its stderr."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class CommandError(RuntimeError):
    """An external command (gh, git, agent) exited non-zero or returned unusable output."""


def run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None = None,
    ok_codes: tuple[int, ...] = (0,),
) -> str:
    """Run ``cmd`` and return stdout; raise :class:`CommandError` on any failure.

    ``ok_codes`` lists exit statuses that are not failures (``git grep`` uses 1
    for "no match").
    """
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandError(f"{shlex.join(cmd)} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise CommandError(f"{cmd[0]}: command not found") from exc
    if proc.returncode not in ok_codes:
        raise CommandError(
            f"{shlex.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def gh(args: Sequence[str], *, cwd: Path) -> str:
    return run(["gh", *args], cwd=cwd)


def gh_json(args: Sequence[str], *, cwd: Path) -> Any:
    out = gh(args, cwd=cwd)
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise CommandError(f"gh {shlex.join(args)} returned non-JSON output") from exc


def gh_list(args: Sequence[str], *, limit: int, cwd: Path) -> list[dict[str, Any]]:
    """Run a ``gh ... list`` command and fail if the result may have been truncated."""
    rows = gh_json([*args, "--limit", str(limit)], cwd=cwd)
    if not isinstance(rows, list):
        raise CommandError(f"gh {shlex.join(args)} did not return a JSON list")
    if len(rows) >= limit:
        raise CommandError(
            f"gh {shlex.join(args)} hit --limit {limit}; results may be truncated"
        )
    return rows


def git(args: Sequence[str], *, cwd: Path, ok_codes: tuple[int, ...] = (0,)) -> str:
    return run(["git", *args], cwd=cwd, ok_codes=ok_codes)
