"""Allowlisted GitHub GETs. Mutable resources carry acquisition and head identity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode

from .retrieval import (
    COMMIT,
    NUMBER,
    PAGE,
    SHA,
    STRING,
    TEXT,
    Retrieval,
    RetrievalError,
    Retrieved,
    Tool,
    pagination,
    strict_json,
    text_page,
)

COMMENT_KINDS = ("discussion", "review_comment", "review")


def object_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RetrievalError("INVALID_SOURCE", "expected a GitHub JSON object")
    return value


def rows_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RetrievalError("INVALID_SOURCE", "expected a GitHub object list")
    return value


def total_value(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise RetrievalError(
            "INVALID_SOURCE", "GitHub total_count is missing or invalid"
        )
    return value


def sha_value(value: Any) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise RetrievalError("INVALID_SOURCE", "GitHub head SHA is missing or invalid")
    return value


def body_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RetrievalError("INVALID_SOURCE", "GitHub body is not text or null")
    return value


def check_diff(text: str, changed_files: int) -> None:
    """Detect omitted files and incomplete unified hunks in the raw diff response."""
    files = 0
    old = new = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if old or new:
                raise RetrievalError(
                    "INCOMPLETE_DIFF", "diff hunk ended before its declared range"
                )
            files += 1
        elif line.startswith("@@ "):
            if old or new:
                raise RetrievalError(
                    "INCOMPLETE_DIFF", "diff hunk ended before its declared range"
                )
            match = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            if match is None:
                raise RetrievalError("INVALID_SOURCE", "malformed unified diff hunk")
            old, new = (
                int(count) if count is not None else 1 for count in match.groups()
            )
        elif old or new:
            if line.startswith(" "):
                old -= 1
                new -= 1
            elif line.startswith("+"):
                new -= 1
            elif line.startswith("-"):
                old -= 1
            elif not line.startswith("\\ No newline at end of file"):
                raise RetrievalError(
                    "INCOMPLETE_DIFF", "unexpected line inside diff hunk"
                )
            if old < 0 or new < 0:
                raise RetrievalError(
                    "INVALID_SOURCE", "diff hunk exceeds its declared range"
                )
    if old or new or files != changed_files:
        raise RetrievalError(
            "INCOMPLETE_DIFF",
            "diff does not cover PR files/hunk ranges",
            expected_files=changed_files,
            received_files=files,
        )


def summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "number",
        "title",
        "state",
        "html_url",
        "created_at",
        "updated_at",
        "user",
        "labels",
        "draft",
        "name",
        "status",
        "conclusion",
        "head_sha",
        "run_id",
        "run_attempt",
        "path",
        "line",
        "commit_id",
        "submitted_at",
        "pull_request",
    )
    # Bodies and nested repository metadata are fetched only on explicit detail calls.
    result = {k: row[k] for k in keys if k in row}
    if "user" in result:
        user = object_value(result["user"])
        result["user"] = {"login": user["login"]}
    if "labels" in result:
        result["labels"] = [object_value(label)["name"] for label in row["labels"]]
    if "pull_request" in result:
        result["pull_request"] = {
            k: row["pull_request"][k]
            for k in ("url", "html_url")
            if k in row["pull_request"]
        }
    return result


@dataclass
class ApiResponse:
    body: bytes
    headers: dict[str, str]
    source: dict[str, Any]

    def json(self) -> Any:
        return strict_json(self.body)


class GitHubReader:
    def __init__(self, retrieval: Retrieval) -> None:
        self.r = retrieval
        self.root = "repos/" + retrieval.repository

    def tools(self) -> list[Tool]:
        kind = {"type": "string", "enum": ["issue", "pr"]}
        comments = {"type": "string", "enum": list(COMMENT_KINDS)}
        return [
            Tool(
                "github_items",
                "Search/list this repository's issues or PRs with a literal phrase and filters. Search's 1000-result ceiling/incomplete responses are errors; narrow filters.",
                self.items,
                {
                    "kind": kind,
                    "text": STRING,
                    "state": {"type": "string", "enum": ["all", "open", "closed"]},
                    "label": STRING,
                    "author": {"type": "string", "pattern": "^[A-Za-z0-9-]+$"},
                    "updated_after": {
                        "type": "string",
                        "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                    },
                    **PAGE,
                },
                ("kind",),
            ),
            Tool(
                "github_item",
                "Read issue/PR metadata and a body range. PR responses include observed base and head SHAs.",
                self.item,
                {"kind": kind, "number": NUMBER, **TEXT},
                ("kind", "number"),
            ),
            Tool(
                "github_comments",
                "Page discussion/review comment or review metadata without injecting bodies. Use github_comment for a selected body range.",
                self.comments,
                {"number": NUMBER, "kind": comments, **PAGE},
                ("number",),
            ),
            Tool(
                "github_comment",
                "Read a selected comment/review body range, verifying its issue/PR association.",
                self.comment,
                {"number": NUMBER, "comment_id": NUMBER, "kind": comments, **TEXT},
                ("number", "comment_id"),
            ),
            Tool(
                "pr_diff",
                "Read PR diff lines with an expected full head SHA; head/base changes during retrieval fail explicitly.",
                self.pr_diff,
                {"number": NUMBER, "head_sha": COMMIT, **TEXT},
                ("number", "head_sha"),
            ),
            Tool(
                "ci_runs",
                "Page workflow runs with branch/status/event/head/date filters. A capped result set requires narrower filters.",
                self.runs,
                {
                    "branch": STRING,
                    "status": {
                        "type": "string",
                        "enum": [
                            "queued",
                            "in_progress",
                            "completed",
                            "success",
                            "failure",
                            "cancelled",
                            "skipped",
                            "timed_out",
                            "action_required",
                            "neutral",
                            "waiting",
                            "pending",
                            "requested",
                        ],
                    },
                    "event": {"type": "string", "pattern": "^[A-Za-z0-9_]+$"},
                    "head_sha": COMMIT,
                    "created_after": {
                        "type": "string",
                        "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                    },
                    **PAGE,
                },
            ),
            Tool(
                "ci_jobs",
                "Page jobs from one captured workflow run attempt, retaining run ID, attempt and code head SHA.",
                self.jobs,
                {"run_id": NUMBER, **PAGE},
                ("run_id",),
            ),
            Tool(
                "ci_log",
                "Read a workflow job log by line/character range, retaining job/run IDs and code head SHA. Missing/expired logs are errors.",
                self.log,
                {"job_id": NUMBER, **TEXT},
                ("job_id",),
            ),
        ]

    def api(
        self,
        endpoint: str,
        *,
        accept: str = "application/vnd.github+json",
        params: dict[str, Any] | None = None,
    ) -> ApiResponse:
        if params:
            endpoint += "?" + urlencode(params)
        source: dict[str, Any] = {
            "kind": "github",
            "method": "GET",
            "url": "https://api.github.com/" + endpoint,
        }
        self.r.acquisitions.append(source)
        try:
            raw = self.r.command.run(
                [
                    "gh",
                    "api",
                    "--hostname",
                    "github.com",
                    "--method",
                    "GET",
                    "--include",
                    "--header",
                    "Accept: " + accept,
                    "--header",
                    "X-GitHub-Api-Version: 2022-11-28",
                    endpoint,
                ]
            )
            headers: dict[str, str] = {}
            # gh includes the final HTTP response headers (also accept redirect headers).
            while raw.startswith(b"HTTP/"):
                boundary = b"\r\n\r\n" if b"\r\n\r\n" in raw else b"\n\n"
                head, separator, raw = raw.partition(boundary)
                if not separator:
                    raise RetrievalError(
                        "INVALID_SOURCE", "GitHub HTTP headers are incomplete"
                    )
                lines = head.decode("utf-8").splitlines()
                status = int(lines[0].split()[1])
                headers = {
                    k.lower(): v.strip()
                    for k, v in (line.split(":", 1) for line in lines[1:])
                }
                if not 200 <= status < 400:
                    raise RetrievalError(
                        "UPSTREAM_ERROR", "GitHub GET failed", http_status=status
                    )
                source["http_status"] = status
            if source.get("http_status") != 200:
                raise RetrievalError(
                    "INVALID_SOURCE", "GitHub did not return a final HTTP 200 response"
                )
            return ApiResponse(raw, headers, source)
        except RetrievalError as exc:
            exc.sources = [source, *exc.sources]
            raise

    def page(
        self,
        response: ApiResponse,
        rows: list[dict[str, Any]],
        page: int,
        per_page: int,
        total: int | None = None,
    ) -> dict[str, Any]:
        if len(rows) > per_page:
            raise RetrievalError(
                "INVALID_SOURCE",
                "GitHub ignored the requested page size",
                sources=[response.source],
            )
        more = bool(re.search(r'rel="next"', response.headers.get("link", "")))
        if total is not None:
            if (page - 1) * per_page > total:
                raise RetrievalError(
                    "RANGE_ERROR",
                    "page exceeds result set",
                    sources=[response.source],
                    total=total,
                )
            expected = page * per_page < total
            if more != expected or (expected and len(rows) < per_page):
                raise RetrievalError(
                    "INCOMPLETE_PAGE",
                    "GitHub pagination disagrees with total_count",
                    sources=[response.source],
                    total=total,
                    page=page,
                )
        if more and not rows:
            raise RetrievalError(
                "INCOMPLETE_PAGE",
                "GitHub returned an empty intermediate page",
                sources=[response.source],
            )
        return pagination(page, per_page, more, total)

    def items(
        self,
        kind: str,
        text: str = "",
        state: str = "all",
        label: str = "",
        author: str | None = None,
        updated_after: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> Retrieved:
        query = f"repo:{self.r.repository} is:{kind}"
        for value in (text, label):
            if any(c in value for c in ('"', "\\", "\r", "\n", "\0")):
                raise RetrievalError(
                    "INVALID_ARGUMENT",
                    "literal search/filter cannot contain quotes, backslashes or control separators",
                )
        if text:
            query += f' "{text}"'
        if label:
            query += f' label:"{label}"'
        if state != "all":
            query += " is:" + state
        if author:
            query += " author:" + author
        if updated_after:
            date.fromisoformat(updated_after)
            query += " updated:>=" + updated_after
        response = self.api(
            "search/issues",
            params={
                "q": query,
                "sort": "updated",
                "order": "asc",
                "page": page,
                "per_page": per_page,
            },
        )
        payload = object_value(response.json())
        if payload["incomplete_results"] is not False:
            raise RetrievalError(
                "INCOMPLETE_SEARCH",
                "GitHub search is incomplete",
                sources=[response.source],
            )
        total = total_value(payload["total_count"])
        if total > 1000:
            raise RetrievalError(
                "UPSTREAM_LIMIT",
                "GitHub search exposes at most 1000 matches; narrow filters",
                sources=[response.source],
                total=total,
                accessible=1000,
            )
        rows = rows_value(payload["items"])
        return Retrieved(
            [summary(row) for row in rows],
            [response.source],
            self.page(response, rows, page, per_page, total),
        )

    def item(self, kind: str, number: int, **ranges: Any) -> Retrieved:
        response = self.api(
            f"{self.root}/{'pulls' if kind == 'pr' else 'issues'}/{number}"
        )
        row = object_value(response.json())
        if row["number"] != number or (kind == "issue" and "pull_request" in row):
            raise RetrievalError(
                "SOURCE_MISMATCH",
                "GitHub returned a different issue/PR kind",
                sources=[response.source],
            )
        source = response.source
        if kind == "pr":
            source.update(
                pr_head_sha=sha_value(row["head"]["sha"]),
                pr_base_sha=sha_value(row["base"]["sha"]),
            )
        data, paging = text_page(body_value(row["body"]), **ranges)
        data["metadata"] = summary(row)
        return Retrieved(data, [source], paging)

    def comment_endpoint(
        self, number: int, kind: str, comment_id: int | None = None
    ) -> str:
        if kind == "review":
            endpoint = f"{self.root}/pulls/{number}/reviews"
        elif comment_id is not None:
            endpoint = (
                f"{self.root}/{'issues' if kind == 'discussion' else 'pulls'}/comments"
            )
        else:
            endpoint = f"{self.root}/{'issues' if kind == 'discussion' else 'pulls'}/{number}/comments"
        return endpoint + (f"/{comment_id}" if comment_id is not None else "")

    def comments(
        self, number: int, kind: str = "discussion", page: int = 1, per_page: int = 50
    ) -> Retrieved:
        response = self.api(
            self.comment_endpoint(number, kind),
            params={"page": page, "per_page": per_page},
        )
        rows = rows_value(response.json())
        return Retrieved(
            [summary(row) for row in rows],
            [response.source],
            self.page(response, rows, page, per_page),
        )

    def comment(
        self, number: int, comment_id: int, kind: str = "discussion", **ranges: Any
    ) -> Retrieved:
        response = self.api(self.comment_endpoint(number, kind, comment_id))
        row = object_value(response.json())
        expected = f"https://api.github.com/{self.root}/{'issues' if kind == 'discussion' else 'pulls'}/{number}"
        association = row.get(
            "issue_url" if kind == "discussion" else "pull_request_url"
        )
        if row["id"] != comment_id or (kind != "review" and association != expected):
            raise RetrievalError(
                "SOURCE_MISMATCH",
                "comment belongs to a different issue/PR",
                sources=[response.source],
            )
        data, paging = text_page(body_value(row["body"]), **ranges)
        data["metadata"] = summary(row)
        return Retrieved(data, [response.source], paging)

    def pr_diff(self, number: int, head_sha: str, **ranges: Any) -> Retrieved:
        endpoint = f"{self.root}/pulls/{number}"
        before = self.api(endpoint)
        pr = object_value(before.json())
        base = sha_value(pr["base"]["sha"])
        observed = sha_value(pr["head"]["sha"])
        if observed != head_sha:
            raise RetrievalError(
                "HEAD_CHANGED",
                "PR head differs from expected SHA",
                sources=[before.source],
                expected=head_sha,
                actual=observed,
            )
        response = self.api(endpoint, accept="application/vnd.github.diff")
        after = self.api(endpoint)
        current = object_value(after.json())
        sources = [before.source, response.source, after.source]
        for source in sources:
            source.update(pr_head_sha=head_sha, pr_base_sha=base)
        if (sha_value(current["head"]["sha"]), sha_value(current["base"]["sha"])) != (
            head_sha,
            base,
        ):
            raise RetrievalError(
                "HEAD_CHANGED",
                "PR head/base changed during diff acquisition",
                sources=sources,
            )
        text = response.body.decode("utf-8")
        if (pr["changed_files"] and not text.startswith("diff --git ")) or pr[
            "changed_files"
        ] != current["changed_files"]:
            raise RetrievalError(
                "INCOMPLETE_DIFF",
                "GitHub diff is absent or inconsistent with PR metadata",
                sources=sources,
            )
        check_diff(text, total_value(pr["changed_files"]))
        data, paging = text_page(text, **ranges)
        return Retrieved(data, sources, paging)

    def runs(
        self,
        branch: str | None = None,
        status: str | None = None,
        event: str | None = None,
        head_sha: str | None = None,
        created_after: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> Retrieved:
        params = {
            k: v
            for k, v in {
                "branch": branch,
                "status": status,
                "event": event,
                "head_sha": head_sha,
                "page": page,
                "per_page": per_page,
            }.items()
            if v is not None
        }
        if created_after:
            date.fromisoformat(created_after)
            params["created"] = ">=" + created_after
        response = self.api(self.root + "/actions/runs", params=params)
        payload = object_value(response.json())
        total = total_value(payload["total_count"])
        if total > 1000:
            raise RetrievalError(
                "UPSTREAM_LIMIT",
                "workflow run result set may exceed GitHub's 1000-result ceiling; narrow filters",
                sources=[response.source],
                total=total,
            )
        rows = rows_value(payload["workflow_runs"])
        for row in rows:
            sha_value(row["head_sha"])
        return Retrieved(
            [summary(row) for row in rows],
            [response.source],
            self.page(response, rows, page, per_page, total),
        )

    def jobs(self, run_id: int, page: int = 1, per_page: int = 50) -> Retrieved:
        run_response = self.api(f"{self.root}/actions/runs/{run_id}")
        run = object_value(run_response.json())
        head = sha_value(run["head_sha"])
        attempt = run["run_attempt"]
        if run["id"] != run_id or type(attempt) is not int or attempt < 1:
            raise RetrievalError(
                "SOURCE_MISMATCH",
                "workflow run identity/attempt mismatch",
                sources=[run_response.source],
            )
        response = self.api(
            f"{self.root}/actions/runs/{run_id}/attempts/{attempt}/jobs",
            params={"page": page, "per_page": per_page},
        )
        payload = object_value(response.json())
        rows = rows_value(payload["jobs"])
        for row in rows:
            if row["run_id"] != run_id or sha_value(row["head_sha"]) != head:
                raise RetrievalError(
                    "SOURCE_MISMATCH",
                    "job belongs to a different run/head",
                    sources=[response.source],
                )
        for source in (run_response.source, response.source):
            source.update(head_sha=head, run_id=run_id, run_attempt=attempt)
        return Retrieved(
            [summary(row) for row in rows],
            [run_response.source, response.source],
            self.page(
                response, rows, page, per_page, total_value(payload["total_count"])
            ),
        )

    def log(self, job_id: int, **ranges: Any) -> Retrieved:
        metadata = self.api(f"{self.root}/actions/jobs/{job_id}")
        row = object_value(metadata.json())
        if row["id"] != job_id:
            raise RetrievalError(
                "SOURCE_MISMATCH", "job identity mismatch", sources=[metadata.source]
            )
        head = sha_value(row["head_sha"])
        response = self.api(
            f"{self.root}/actions/jobs/{job_id}/logs", accept="text/plain"
        )
        if not response.body:
            raise RetrievalError(
                "EMPTY_LOG",
                "GitHub returned no job log content",
                sources=[response.source],
            )
        for source in (metadata.source, response.source):
            source.update(head_sha=head, job_id=job_id, run_id=row["run_id"])
        data, paging = text_page(response.body.decode("utf-8"), **ranges)
        data["metadata"] = summary(row)
        return Retrieved(data, [metadata.source, response.source], paging)
