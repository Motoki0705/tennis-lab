"""GitHub I/O. Reuse the shared issues reader; writes use structured JSON stdin."""

from __future__ import annotations

import copy
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

from issues.cli import GitHub as IssueGitHub

from . import metrics
from .policy import closing_issues, object_value

if TYPE_CHECKING:
    from .state import Store


class GitHub(IssueGitHub):
    def __init__(self, repo: str | None, *, store: Store | None = None) -> None:
        super().__init__(repo)
        self._store = store
        self.parent: int | None = None
        self._cache: dict[tuple[str, bool], object] = {}
        self._fresh = False
        self._cache_epoch = -1

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        if (
            self._store is not None
            and self._cache_epoch != self._store.transaction_epoch
        ):
            self._cache.clear()
            self._cache_epoch = self._store.transaction_epoch
        key = (endpoint, missing_parent)
        if (
            self._store is not None
            and self.parent is not None
            and not self._fresh
            and key in self._cache
        ):
            metrics.count("cache_hits")
            return copy.deepcopy(self._cache[key])
        with self._store.unlocked_read() if self._store else nullcontext():
            started = time.monotonic()
            metrics.count("github_requests")
            if "page=" in endpoint:
                metrics.count("github_pages")
            try:
                result = super().get(endpoint, missing_parent=missing_parent)
            finally:
                metrics.count("github_seconds", time.monotonic() - started)
        self._cache[key] = copy.deepcopy(result)
        if "/sub_issues?" in endpoint and isinstance(result, list):
            for issue in result:
                if (
                    isinstance(issue, dict)
                    and {"number", "body", "title", "labels", "state"} <= issue.keys()
                ):
                    self.identity(issue)
                    self._cache[
                        (f"repos/{self.repo}/issues/{issue['number']}", False)
                    ] = copy.deepcopy(issue)
        return result

    @contextmanager
    def fresh_reads(self) -> Iterator[None]:
        previous, self._fresh = self._fresh, True
        try:
            yield
        finally:
            self._fresh = previous

    def invalidate(self, endpoint: str) -> None:
        # Writes invalidate the issue and its relations, retaining unrelated reads.
        root = endpoint.split("/comments")[0].split("/sub_issues")[0]
        self._cache = {
            key: value
            for key, value in self._cache.items()
            if not key[0].split("?")[0].startswith(root)
        }

    def rows(self, endpoint: str) -> list[dict[str, Any]]:
        for attempt in range(3):
            result: list[dict[str, Any]] = []
            seen: set[object] = set()
            page = 1
            with self.fresh_reads() if attempt else nullcontext():
                while True:
                    separator = "&" if "?" in endpoint else "?"
                    value = self.get(f"{endpoint}{separator}per_page=100&page={page}")
                    if not isinstance(value, list) or len(value) > 100:
                        raise ValueError("expected a GitHub page of at most 100 rows")
                    rows = [object_value(row) for row in value]
                    keys = []
                    for row in rows:
                        identity = row.get("id", row.get("number"))
                        if identity is None and row.get("event") == "cross-referenced":
                            source = object_value(
                                object_value(row.get("source")).get("issue")
                            )
                            if source.get("id") is not None and row.get("created_at"):
                                identity = (
                                    "cross-referenced",
                                    source["id"],
                                    row["created_at"],
                                )
                        if identity is None and row.get("sha"):
                            identity = (row.get("event"), row["sha"])
                        keys.append(identity)
                    if any(key is None for key in keys):
                        raise ValueError("GitHub pagination row has no stable identity")
                    if len(set(keys)) != len(keys) or seen.intersection(keys):
                        metrics.count("pagination_retries")
                        break
                    seen.update(keys)
                    result.extend(rows)
                    if len(rows) < 100:
                        return result
                    page += 1
        raise ValueError(
            "GitHub pages remained inconsistent after three bounded attempts"
        )

    def write(
        self, method: str, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.invalidate(endpoint)
        with self._store.unlocked_read() if self._store else nullcontext():
            started = time.monotonic()
            metrics.count("github_requests")
            try:
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
            finally:
                metrics.count("github_seconds", time.monotonic() - started)
        if result.returncode:
            raise ValueError(f"gh {method} {endpoint} failed: {result.stderr.strip()}")
        return object_value(json.loads(result.stdout))

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Read GraphQL without caching or holding the registry lock remotely."""
        with self._store.unlocked_read() if self._store else nullcontext():
            started = time.monotonic()
            metrics.count("github_requests")
            try:
                result = subprocess.run(
                    ["gh", "api", "graphql", "--input", "-"],
                    input=json.dumps({"query": query, "variables": variables}),
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
            finally:
                metrics.count("github_seconds", time.monotonic() - started)
        if result.returncode:
            raise ValueError(f"gh graphql failed: {result.stderr.strip()}")
        response = object_value(json.loads(result.stdout))
        if response.get("errors"):
            raise ValueError(f"gh graphql errors: {response['errors']}")
        data: dict[str, Any] = object_value(response.get("data"))
        return data

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

    def pulls(self, issues: set[int] | None = None) -> list[dict[str, Any]]:
        if self.parent is None and issues is None:
            return self.rows(f"repos/{self.repo}/pulls?state=all")
        # Timeline cross-references find PRs even before result registration.
        # Fetch PRs by immutable number; changing PR sort order cannot duplicate
        # or omit records across pages of a repository-wide pulls listing.
        numbers: set[int] = set()
        if issues is None:
            assert self.parent is not None
            family = set(self.family(self.parent))
        else:
            family = issues
        for number in family:
            for event in self.rows(f"repos/{self.repo}/issues/{number}/timeline"):
                source = event.get("source", {}).get("issue", {})
                if not source.get("pull_request"):
                    continue
                if "body" not in source:
                    raise ValueError("GitHub timeline PR omitted its body")
                if not closing_issues(source.get("body") or "", self.repo) & family:
                    continue
                url = source.get("repository_url", "")
                if (
                    url.casefold()
                    != f"https://api.github.com/repos/{self.repo}".casefold()
                ):
                    continue
                n = source.get("number")
                if type(n) is not int or n < 1:
                    raise ValueError("invalid PR timeline reference")
                numbers.add(n)
        result = []
        for n in sorted(numbers):
            row = object_value(self.get(f"repos/{self.repo}/pulls/{n}"))
            if row.get("number") != n:
                raise ValueError("GitHub PR identity mismatch")
            result.append(row)
        return result
