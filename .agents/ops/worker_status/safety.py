"""Apply the board public contract before publishing any free-form runner text."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from board.errors import BoardError
from board.safety import public


class StatusError(ValueError):
    """A public error code, without private paths, log excerpts or identities."""


class Guard:
    def __init__(self, parents: Mapping[str, Any]) -> None:
        self.parents = parents
        self.env = dict(os.environ)
        self.private: set[str] = set()
        # Logs can quote arbitrary environment values. Do not expose them just
        # because their variable name is not one of the board's identity keys.
        self.private.update(v for v in self.env.values() if len(v) >= 8)
        self.observe_parents(parents)

    def observe_parents(self, parents: Mapping[str, Any]) -> None:
        """Keep all observed owners/run references private across a handoff retry."""
        for row in parents.values():
            self.remember(row["session"].partition(":")[2])
            for task in row["tasks"].values():
                self.remember(task.get("delegation_id"))
                self.remember_path(task.get("worktree"))
                runs = task.get("worker_runs", [])
                for run in runs if isinstance(runs, list) else []:
                    if isinstance(run, dict):
                        self.remember_path(run.get("run_dir"))
                        self.remember(run.get("delegation_id"))

    def remember(self, *values: Any) -> None:
        self.private.update(v for v in values if isinstance(v, str) and v)

    def remember_path(self, value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        name = Path(value).name
        if "/" in value or "\\" in value:
            self.private.add(value)
        if len(name) >= 8 and name != "registered":
            self.private.add(name)

    def check(self, value: Any) -> None:
        try:
            public(value, self.parents, self.env)
        except BoardError as exc:
            raise StatusError("unsafe_public_field") from exc

        def visit(item: Any) -> None:
            if isinstance(item, str):
                if any(secret in item for secret in self.private) or re.search(
                    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,})\b"
                    r"|\b(?:token|password|secret|api[_-]?key)\s*[:=]",
                    item,
                    re.IGNORECASE,
                ):
                    raise StatusError("unsafe_public_field")
            elif isinstance(item, Mapping):
                for key, entry in item.items():
                    visit(key)
                    visit(entry)
            elif isinstance(item, (list, tuple)):
                for entry in item:
                    visit(entry)

        visit(value)

    def text(
        self, value: Any, *, limit: int = 300, first_line: bool = False
    ) -> dict[str, Any]:
        if value is None:
            return {"status": "unavailable", "text": None, "truncated": False}
        if not isinstance(value, str):
            raise ValueError("invalid free text")
        try:
            self.check(value)
        except StatusError:
            return {"status": "hidden", "text": None, "truncated": False}
        shown = value.split("\n", 1)[0] if first_line else value
        return {
            "status": "visible",
            "text": shown[:limit],
            "truncated": len(shown) > limit,
        }
