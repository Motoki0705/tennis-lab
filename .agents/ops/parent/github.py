"""GitHub I/O with a shared HTTPS pool and explicit transaction boundaries."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues.cli import GitHub as IssueGitHub
from issues.evidence import EVIDENCE_LABEL, is_evidence_issue, require_work_issue

from . import metrics
from .policy import closing_issues, object_value

if TYPE_CHECKING:
    from .evidence import EvidenceIssues
    from .history import History
    from .state import Store


class GitHub(IssueGitHub):
    def improvement_extractions(self, store: Store) -> dict[str, Any]:
        from .improvement_receipts import extraction_index

        return extraction_index(store.directory, self.repo)

    def __init__(self, repo: str | None, *, store: Store | None = None) -> None:
        super().__init__(repo)
        self._store = store
        self._outside_store: Store | None = None
        self.parent: int | None = None
        self._cache: dict[tuple[str, bool], object] = {}
        self._fresh = False
        self._cache_epoch = -1
        self._history_rows: list[dict[str, Any]] | None = None
        self._history_scope = False
        self.evidence_directory: Path | None = (
            store.directory / "evidence-issues" if store else None
        )

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        self._require_scope()
        if (
            self._store is not None
            and self._cache_epoch != self._store.transaction_epoch
        ):
            self._cache.clear()
            self._history_rows = None
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
        with self.io_scope():
            started = time.monotonic()
            metrics.count("github_requests")
            if "page=" in endpoint:
                metrics.count("github_pages")
            try:
                result = super().get(endpoint, missing_parent=missing_parent)
            finally:
                metrics.count("github_seconds", time.monotonic() - started)
        if self._store is not None:
            self._cache[key] = copy.deepcopy(result)
        if (
            self._store is not None
            and "/sub_issues?" in endpoint
            and isinstance(result, list)
        ):
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

    def _require_scope(self) -> None:
        if self._store is not None:
            self._store.require_transaction()
        elif self._outside_store is not None:
            self._outside_store.require_unlocked()

    @contextmanager
    def io_scope(self) -> Iterator[None]:
        """Release only this client's transaction, never infer a missing lock."""
        self._require_scope()
        with self._store.unlocked_read() if self._store else nullcontext():
            yield

    def detached(self) -> GitHub:
        """Explicit registry-external client without an endpoint GET cache."""
        client = copy.copy(self)
        client._outside_store = self._store or self._outside_store
        client._store = None
        client._cache = {}
        client._cache_epoch = -1
        client._history_rows = None
        client._history_scope = False
        client._fresh = False
        return client

    @contextmanager
    def fresh_reads(self) -> Iterator[None]:
        previous, self._fresh = self._fresh, True
        # Begin one new observation. Repeated history reads within this scope
        # share its discovery index; endpoint GETs still bypass their cache.
        self._history_rows = None
        try:
            yield
        finally:
            self._fresh = previous

    def invalidate(self, endpoint: str) -> None:
        # Writes invalidate the issue and its relations, retaining unrelated reads.
        self._history_rows = None
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
        self._require_scope()
        self.invalidate(endpoint)
        with self.io_scope():
            started = time.monotonic()
            metrics.count("github_requests")
            try:
                result = self._http.request(method, endpoint, payload)
            finally:
                metrics.count("github_seconds", time.monotonic() - started)
        if result.returncode:
            raise ValueError(
                f"gh {method} {self._http.redact(endpoint)} failed: {self._http.redact(result.stderr.strip())}"
            )
        return object_value(json.loads(self._http.redact_json(result.stdout)))

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Read GraphQL without caching or holding the registry lock remotely."""
        with self.io_scope():
            started = time.monotonic()
            metrics.count("github_requests")
            try:
                result = self._http.request(
                    "POST", "graphql", {"query": query, "variables": variables}
                )
            finally:
                metrics.count("github_seconds", time.monotonic() - started)
        if result.returncode:
            raise ValueError(
                f"gh graphql failed: {self._http.redact(result.stderr.strip())}"
            )
        response = object_value(json.loads(self._http.redact_json(result.stdout)))
        if response.get("errors"):
            raise ValueError(f"gh graphql errors: {response['errors']}")
        data: dict[str, Any] = object_value(response.get("data"))
        return data

    def issue(self, number: int, *, allow_proposal: bool = False) -> dict[str, Any]:
        row = object_value(self.get(f"repos/{self.repo}/issues/{number}"))
        if self.identity(row) != number:
            raise ValueError("GitHub issue identity mismatch")
        require_work_issue(row, allow_proposal=allow_proposal)
        return row

    def issue_rows(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.rows(f"repos/{self.repo}/issues?state=all")
            if "pull_request" not in row and not is_evidence_issue(row)
        ]

    @contextmanager
    def history_read_scope(self) -> Iterator[None]:
        """One complete discovery index per read-only collection, never across observations."""
        self._require_scope()
        self._cache.clear()
        previous = getattr(self, "_history_rows", None)
        active = getattr(self, "_history_scope", False)
        self._history_rows = None
        self._history_scope = True
        try:
            yield
        finally:
            self._history_rows, self._history_scope = previous, active

    def evidence_rows(self) -> list[dict[str, Any]]:
        self._require_scope()
        endpoint = f"repos/{self.repo}/issues?state=all&labels={EVIDENCE_LABEL}"
        if not getattr(self, "_history_scope", False):
            return self.rows(endpoint)
        if self._history_rows is None:
            # A new scope, explicit freshness check or write invalidates discovery.
            # Do not resurrect an older listing from the transaction GET cache.
            with self.fresh_reads():
                self._history_rows = self.rows(endpoint)
        return self._history_rows

    def evidence_issues(self) -> EvidenceIssues:
        from .evidence import EvidenceIssues
        from .state import state_directory

        directory = getattr(self, "evidence_directory", None)
        if directory is None:
            directory = (
                self._store.directory if self._store else state_directory()
            ) / "evidence-issues"
        return EvidenceIssues(self, directory)

    def record_history(
        self, parent: int, login: str | None = None, **kwargs: Any
    ) -> History:
        from .history import read_history

        publisher = login or object_value(self.get("user"))["login"]
        return read_history(self, parent, publisher, **kwargs)

    def parent_comments(
        self, parent: int, login: str | None = None
    ) -> list[dict[str, Any]]:
        """Validated compatibility projection; raw comments() remains a single medium."""
        return self.record_history(parent, login).comments

    def review_comments(
        self, issue: int, parent: int, login: str
    ) -> list[dict[str, Any]]:
        """PR projections stay on the PR; parent-side evidence uses the shared history."""
        return (
            self.parent_comments(parent, login)
            if issue == parent
            else self.comments(issue)
        )

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
