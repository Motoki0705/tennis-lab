"""Subprocess helpers: every external command fails loudly with its stderr.

``run`` / ``run_process_group`` / ``CommandError`` live in ``.agents/ops/lib/opsproc.py``
(shared with the other ops jobs); this module adds the gh/git conveniences.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

from opsproc import CommandError, run, run_process_group  # noqa: E402

__all__ = ["CommandError", "gh", "gh_json", "gh_list", "git", "run", "run_process_group"]


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
