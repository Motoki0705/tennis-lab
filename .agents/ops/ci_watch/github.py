"""The only network boundary: explicit REST GETs through gh, with raw evidence."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .store import Store

SHA = re.compile(r"[0-9a-f]{40}\Z")


class FetchError(RuntimeError):
    def __init__(self, endpoint: str, reason: str) -> None:
        self.endpoint = endpoint
        self.reason = reason
        super().__init__(f"{endpoint}: {reason}")

    def record(self) -> dict[str, str]:
        return {"endpoint": self.endpoint, "reason": self.reason}


def sha(value: Any) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"expected a full commit SHA, got {value!r}")
    return value


def positive_int(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"expected a positive integer, got {value!r}")
    return value


def required(row: Any, *fields: str) -> dict[str, Any]:
    if not isinstance(row, dict) or any(field not in row for field in fields):
        raise ValueError(f"expected an object with {fields}")
    return row


class GitHub:
    def __init__(self, repository: str, store: Store, cwd: Path) -> None:
        self.prefix = f"repos/{repository}"
        self.store = store
        self.cwd = cwd

    def get(self, endpoint: str) -> tuple[str, dict[str, str]]:
        if not endpoint.startswith(self.prefix + "/"):
            raise ValueError("endpoint is outside the configured repository")
        try:
            result = subprocess.run(
                ["gh", "api", "--hostname", "github.com", "--method", "GET", endpoint],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
        except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
            raise FetchError(endpoint, str(exc)) from exc
        if result.returncode:
            raise FetchError(
                endpoint, result.stderr.strip() or f"gh exit {result.returncode}"
            )
        ref = {"endpoint": endpoint, "object": self.store.put_object(result.stdout)}
        return result.stdout, ref

    def json(self, endpoint: str) -> tuple[Any, dict[str, str]]:
        body, ref = self.get(endpoint)
        try:
            return json.loads(body), ref
        except ValueError as exc:
            raise FetchError(
                endpoint, f"invalid JSON; evidence {ref['object']}"
            ) from exc

    def pages(
        self, path: str, key: str, **query: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Check advertised totals; never accept a search cap or a partial page."""
        rows: list[dict[str, Any]] = []
        refs = []
        ids: set[int] = set()
        page = 1
        total = None
        while True:
            endpoint = f"{self.prefix}/{path}?{urlencode({**query, 'per_page': 100, 'page': page})}"
            body, ref = self.json(endpoint)
            try:
                required(body, "total_count", key)
                count = body["total_count"]
                if (
                    type(count) is not int
                    or count < 0
                    or (total is not None and total != count)
                ):
                    raise ValueError("pagination total changed; retry the collection")
                total = count
                if key == "workflow_runs" and total >= 1000:
                    raise ValueError(
                        "workflow run search reached the API's 1000-result cap"
                    )
                batch = body[key]
                if not isinstance(batch, list) or len(batch) > 100:
                    raise ValueError("invalid page")
                for row in batch:
                    identifier = positive_int(required(row, "id")["id"])
                    if identifier in ids:
                        raise ValueError(
                            "duplicate ID across pages; retry the collection"
                        )
                    ids.add(identifier)
                rows.extend(batch)
                refs.append(ref)
                if len(rows) == total:
                    return rows, refs
                if len(batch) < 100 or len(rows) > total:
                    raise ValueError(
                        "incomplete pagination (possibly the API's search cap)"
                    )
            except (ValueError, TypeError, KeyError) as exc:
                raise FetchError(endpoint, str(exc)) from exc
            page += 1
