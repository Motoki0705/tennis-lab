"""GitHub I/O. Reuse the shared issues reader; writes use structured JSON stdin."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

from issues.cli import GitHub as IssueGitHub

from .policy import object_value

if TYPE_CHECKING:
    from .state import Store


class GitHub(IssueGitHub):
    def __init__(self, repo: str | None, *, store: Store | None = None) -> None:
        super().__init__(repo)
        self._store = store

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        if self._store is None:
            return super().get(endpoint, missing_parent=missing_parent)
        with self._store.unlocked_read():
            return super().get(endpoint, missing_parent=missing_parent)

    def rows(self, endpoint: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        page = 1
        while True:
            separator = "&" if "?" in endpoint else "?"
            value = self.get(f"{endpoint}{separator}per_page=100&page={page}")
            if not isinstance(value, list) or len(value) > 100:
                raise ValueError("expected a GitHub page of at most 100 rows")
            result.extend(object_value(row) for row in value)
            if len(value) < 100:
                return result
            page += 1

    def write(
        self, method: str, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                method,
                "--header",
                "Accept: application/vnd.github+json",
                "--header",
                "X-GitHub-Api-Version: 2022-11-28",
                endpoint,
                "--input",
                "-",
            ],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"gh {method} {endpoint} failed: {result.stderr.strip()}")
        return object_value(json.loads(result.stdout))

    def issue(self, number: int) -> dict[str, Any]:
        row = object_value(self.get(f"repos/{self.repo}/issues/{number}"))
        if self.identity(row) != number:
            raise ValueError("GitHub issue identity mismatch")
        return row

    def issue_rows(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.rows(f"repos/{self.repo}/issues?state=all")
            if "pull_request" not in row
        ]

    def comments(self, parent: int) -> list[dict[str, Any]]:
        return self.rows(f"repos/{self.repo}/issues/{parent}/comments")

    def pulls(self) -> list[dict[str, Any]]:
        return self.rows(f"repos/{self.repo}/pulls?state=all")
