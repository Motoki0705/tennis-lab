"""Read-only GitHub adapter and local draft renderer. No GitHub mutation commands."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .consistency import Issue, check_consistency
from .evidence import EVIDENCE_LABEL, require_work_issue
from .format import (
    CHILD_LABEL,
    PARENT_LABEL,
    PROPOSAL_LABEL,
    STANDALONE_LABEL,
    ChildTask,
    ParentRef,
    ProposalSource,
    TaskStatus,
    child_title,
    parent_title,
    render_child_body,
    render_parent_body,
)
from .polling import GETResponse, JobLog, RateLimit, rate_limit
from .transport import HTTPClient, HTTPResponse

_HTTP_STATUS = re.compile(r"HTTP/\S+ ([0-9]{3})(?:\s|$)")
_GET_RETRY_DELAYS = (1.0, 2.0)


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field}: expected an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected a string")
    return value


def _positive(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field}: expected a positive integer")
    return value


def _keys(data: dict[str, object], allowed: set[str]) -> None:
    extra = data.keys() - allowed
    if extra:
        raise ValueError(f"unknown fields: {sorted(extra)}")


def render_spec(value: object) -> dict[str, object]:
    data = _object(value, "spec")
    _keys(
        data,
        {
            "role",
            "identifier",
            "summary",
            "parent",
            "sections",
            "children",
            "source",
            "labels",
        },
    )
    role = data.get("role")
    if role not in ("parent", "child"):
        raise ValueError("role must be parent or child")
    parent = None
    if data.get("parent") is not None:
        ref = _object(data["parent"], "parent")
        _keys(ref, {"identifier", "number"})
        parent = ParentRef(
            _string(ref.get("identifier"), "parent.identifier"),
            _positive(ref.get("number"), "parent.number"),
        )
    source = None
    if data.get("source") is not None:
        raw = _object(data["source"], "source")
        _keys(raw, {"proposal_id", "report_number", "option"})
        option = raw.get("option")
        source = ProposalSource(
            _string(raw.get("proposal_id"), "source.proposal_id"),
            _positive(raw.get("report_number"), "source.report_number"),
            None if option is None else _string(option, "source.option"),
        )
    sections = {
        key: _string(text, f"sections.{key}")
        for key, text in _object(data.get("sections"), "sections").items()
    }
    summary = _string(data.get("summary"), "summary")
    labels = data.get("labels", [])
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("labels must be a list of nonempty strings")
    if (
        PARENT_LABEL in labels
        or CHILD_LABEL in labels
        or STANDALONE_LABEL in labels
        or PROPOSAL_LABEL in labels
        or EVIDENCE_LABEL in labels
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            "labels must contain unique work-type/area/origin labels; role labels are generated"
        )
    if role == "parent":
        raw_tasks = data.get("children", [])
        if not isinstance(raw_tasks, list):
            raise ValueError("children must be a list")
        tasks: list[ChildTask] = []
        for item in raw_tasks:
            task = _object(item, "child task")
            _keys(task, {"number", "title", "status"})
            status = _string(task.get("status", "open"), "child task status")
            tasks.append(
                ChildTask(
                    _positive(task.get("number"), "child task number"),
                    _string(task.get("title"), "child task title"),
                    cast(TaskStatus, status),
                )
            )
        title = parent_title(
            _string(data.get("identifier"), "identifier"), summary, parent=parent
        )
        body = render_parent_body(
            sections,
            tasks,
            parent_number=parent.number if parent else None,
            source=source,
        )
        role_labels = [PARENT_LABEL, *([CHILD_LABEL] if parent else [])]
    else:
        if parent is None or "identifier" in data or "children" in data:
            raise ValueError(
                "child requires parent and cannot have identifier/children; use role=parent for a nested parent"
            )
        title = child_title(parent, summary)
        body = render_child_body(parent.number, sections, source=source)
        role_labels = [CHILD_LABEL]
    return {
        "dry_run": True,
        "title": title,
        "body": body,
        "labels": sorted([*role_labels, *labels]),
    }


def _repo_name(value: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", value
    ) or value.split("/")[-1] in {".", ".."}:
        raise ValueError("repo must be owner/repo")
    return value


def local_repository() -> str:
    """Resolve every effective fetch/push URL without contacting GitHub."""

    def git_lines(*args: str) -> list[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError(
                "cannot resolve a GitHub remote; specify --repo owner/repo"
            ) from None
        if result.returncode or not result.stdout.strip():
            raise ValueError(
                "cannot resolve a GitHub remote; specify --repo owner/repo"
            )
        return result.stdout.splitlines()

    urls = []
    for remote in git_lines("remote"):
        # Ask git to apply insteadOf/pushInsteadOf, including the fallback to
        # fetch URLs when pushurl is absent. Inspect every URL, not just origin
        # or the first destination, so a rewritten push cannot pick another repo.
        urls.extend(git_lines("remote", "get-url", "--all", "--", remote))
        urls.extend(git_lines("remote", "get-url", "--push", "--all", "--", remote))
    repos: dict[str, str] = {}
    for url in urls:
        scp = re.fullmatch(r"git@github\.com:([^?#]+)", url, re.IGNORECASE)
        if scp:
            path = scp[1]
        else:
            try:
                parsed = urlsplit(url)
                github = (
                    parsed.scheme in {"https", "ssh"}
                    and parsed.hostname is not None
                    and parsed.hostname.casefold() == "github.com"
                    and parsed.port in {None, 22 if parsed.scheme == "ssh" else 443}
                    and not parsed.query
                    and not parsed.fragment
                    and not parsed.password
                )
            except ValueError:
                github = False
            if not github:
                raise ValueError("non-GitHub git remote; specify --repo owner/repo")
            path = parsed.path.removeprefix("/")
        name = _repo_name(path.removesuffix(".git"))
        repos.setdefault(name.casefold(), name)
    if len(repos) != 1:
        raise ValueError("ambiguous GitHub remotes; specify --repo owner/repo")
    return next(iter(repos.values()))


class GitHubHTTPError(ValueError):
    """A failed GET response with its actual HTTP status preserved."""

    def __init__(
        self, endpoint: str, status: int, detail: str, *, attempts: int = 1
    ) -> None:
        self.status = status
        self.attempts = attempts
        super().__init__(
            f"gh api GET {endpoint}: HTTP {status}; attempts={attempts}; "
            f"last_error_kind=http_{status} ({detail})"
        )


class GitHubTransportError(ValueError):
    """A GET exhausted its bounded retries without an HTTP response status."""

    def __init__(
        self, endpoint: str, attempts: int, error_kind: str, detail: str
    ) -> None:
        self.endpoint = endpoint
        self.attempts = attempts
        self.error_kind = error_kind
        super().__init__(
            f"gh api GET {endpoint}: attempts={attempts}; "
            f"last_error_kind={error_kind} ({detail})"
        )


class GitHub:
    def __init__(self, repo: str | None) -> None:
        self.repo = _repo_name(repo) if repo is not None else local_repository()
        self._http = HTTPClient()
        self._conditional_cache: dict[tuple[str, str], tuple[str, object]] = {}
        self._secondary_limits: dict[str, int] = {}

    def close(self) -> None:
        self._http.close()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Scripted GET injection boundary retained for existing callers/tests."""
        return self._http.request("GET", args[-1], include=True)

    def _get_response(
        self, endpoint: str
    ) -> tuple[subprocess.CompletedProcess[str], int]:
        # Only REST GET reaches this loop. An HTTP response ends transport
        # retries even for auth/rate-limit/server errors or malformed bodies.
        for attempt, delay in enumerate((0.0, *_GET_RETRY_DELAYS), start=1):
            if delay:
                time.sleep(delay)
            try:
                result = self._run(
                    "api",
                    "--include",
                    "--method",
                    "GET",
                    "--header",
                    "Accept: application/vnd.github+json",
                    "--header",
                    "X-GitHub-Api-Version: 2022-11-28",
                    endpoint,
                )
            except subprocess.TimeoutExpired as exc:
                partial: str | bytes = exc.stdout or b""
                if isinstance(partial, bytes):
                    partial = partial.decode("utf-8", errors="replace")
                status = _HTTP_STATUS.match(partial)
                if status:
                    raise GitHubHTTPError(
                        self._http.redact(endpoint),
                        int(status[1]),
                        f"timeout after {exc.timeout}s after HTTP response status",
                        attempts=attempt,
                    ) from exc
                error_kind = "timeout_without_http_response"
                detail = f"timeout after {exc.timeout}s without HTTP response status"
            else:
                if _HTTP_STATUS.match(result.stdout):
                    return result, attempt
                if result.returncode == 0:
                    raise ValueError(
                        f"gh api GET {self._http.redact(endpoint)}: attempts={attempt}; "
                        "last_error_kind=invalid_http_response "
                        "(gh exited successfully without HTTP response status)"
                    )
                error_kind = "missing_http_response_status"
                detail = f"missing HTTP response status ({result.stderr.strip()})"
        raise GitHubTransportError(
            self._http.redact(endpoint), attempt, error_kind, self._http.redact(detail)
        )

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        result, attempts = self._get_response(endpoint)
        headers, separator, body = result.stdout.replace("\r\n", "\n").partition("\n\n")
        status = _HTTP_STATUS.match(headers)
        assert status is not None  # _get_response requires an HTTP status.
        if status[1] == "404" and result.returncode != 0 and missing_parent:
            try:
                error = json.loads(body)
            except json.JSONDecodeError:
                error = None
            if (
                isinstance(error, dict)
                and error.get("message") == "No parent issue found"
            ):
                return None
        if result.returncode or status[1] != "200":
            raise GitHubHTTPError(
                self._http.redact(endpoint),
                int(status[1]),
                self._http.redact(result.stderr.strip()),
                attempts=attempts,
            )
        context = f"gh api GET {self._http.redact(endpoint)}: attempts={attempts};"
        if not separator:
            raise ValueError(f"{context} last_error_kind=missing_http_body_separator")
        try:
            payload: object = json.loads(self._http.redact_json(body))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{context} last_error_kind=invalid_json_response ({exc})"
            ) from exc
        if payload is None:
            raise ValueError(f"{context} last_error_kind=unexpected_null_response")
        return payload

    def _poll_response(self, endpoint: str, headers: dict[str, str]) -> HTTPResponse:
        """A polling request never sleeps or retries inside the acquisition layer."""
        try:
            response = self._http.request_response("GET", endpoint, headers=headers)
        except subprocess.TimeoutExpired:
            raise GitHubTransportError(
                self._http.redact(endpoint),
                1,
                "timeout_without_http_response",
                "GitHub polling request timed out before HTTP status",
            ) from None
        if response.status is None:
            raise GitHubTransportError(
                self._http.redact(endpoint),
                1,
                "missing_http_response_status",
                response.error,
            )
        if response.error:
            raise GitHubHTTPError(
                self._http.redact(endpoint), response.status, response.error
            )
        return response

    def _poll_limit(
        self, response: HTTPResponse, principal: str, clock: Callable[[], float]
    ) -> RateLimit | None:
        limit = rate_limit(
            response,
            now=clock(),
            secondary_attempt=self._secondary_limits.get(principal, 0) + 1,
        )
        if limit is not None and limit.kind == "secondary":
            self._secondary_limits[principal] = limit.secondary_attempt
        elif response.status in {200, 304}:
            self._secondary_limits.pop(principal, None)
        return limit

    def get_conditional(
        self, endpoint: str, *, clock: Callable[[], float] = time.time
    ) -> GETResponse:
        """Opt-in ETag GET. Rate limits return deadlines; other failures raise."""
        from parent import metrics

        key = self._http.cache_key(endpoint)
        cached = self._conditional_cache.get(key)
        response = self._poll_response(
            endpoint, {"If-None-Match": cached[0]} if cached else {}
        )
        assert response.status is not None
        limit = self._poll_limit(response, key[1], clock)
        if limit is not None:
            return GETResponse(
                response.status,
                response.headers,
                response.body,
                True if cached else None,
                limit,
            )
        if response.status == 304:
            if cached is None:
                raise ValueError(
                    "GitHub 304 has no saved response for this URL and authentication"
                )
            cached_etag = response.headers.get("etag") or cached[0]
            self._conditional_cache[key] = (cached_etag, cached[1])
            return GETResponse(304, response.headers, copy.deepcopy(cached[1]), True)
        if response.status != 200:
            raise GitHubHTTPError(
                self._http.redact(endpoint), response.status, response.http_error()
            )
        try:
            payload: object = json.loads(response.body)
        except ValueError:
            raise ValueError("GitHub conditional GET returned invalid JSON") from None
        if payload is None:
            raise ValueError("GitHub conditional GET returned unexpected null")
        etag = response.headers.get("etag")
        if etag:
            self._conditional_cache[key] = (etag, copy.deepcopy(payload))
        else:
            self._conditional_cache.pop(key, None)
            metrics.count("github_get_without_etag")
        return GETResponse(200, response.headers, payload, bool(etag))

    def job_log(self, job_id: int, *, clock: Callable[[], float] = time.time) -> JobLog:
        """Resolve the job log redirect once; report missing/expired logs explicitly."""
        if type(job_id) is not int or job_id < 1:
            raise ValueError("job_id must be a positive integer")
        endpoint = f"repos/{self.repo}/actions/jobs/{job_id}/logs"
        try:
            response = self._poll_response(endpoint, {})
        except (GitHubHTTPError, GitHubTransportError) as exc:
            return JobLog(getattr(exc, "status", None), None, str(exc))
        key = self._http.cache_key(endpoint)
        try:
            limit = self._poll_limit(response, key[1], clock)
        except ValueError as exc:
            return JobLog(response.status, None, str(exc))
        if limit is not None:
            return JobLog(response.status, None, f"{limit.kind}_rate_limit", limit)
        if response.status != 302:
            return JobLog(
                response.status,
                None,
                "GitHub job log expected HTTP 302; " + response.http_error(),
            )
        location = response.headers.get("location")
        if not location:
            return JobLog(302, None, "GitHub job log redirect has no Location header")
        try:
            downloaded = self._http.download_log(location)
        except subprocess.TimeoutExpired:
            return JobLog(None, None, "GitHub job log download timed out")
        except ValueError as exc:
            return JobLog(302, None, str(exc))
        if downloaded.error:
            return JobLog(downloaded.status, None, downloaded.error)
        if downloaded.status != 200:
            return JobLog(
                downloaded.status,
                None,
                f"GitHub job log download HTTP {downloaded.status}",
            )
        if not downloaded.body:
            return JobLog(200, None, "GitHub job log download returned an empty body")
        self._secondary_limits.pop(key[1], None)
        return JobLog(200, downloaded.body)

    def identity(self, value: object) -> int:
        issue = _object(value, "issue")
        repository_url = _string(issue.get("repository_url"), "repository_url")
        if urlsplit(repository_url).path.casefold() != f"/repos/{self.repo}".casefold():
            raise ValueError(
                "cross-repository sub-issues are outside this contract; inspect them explicitly"
            )
        if issue.get("pull_request") is not None:
            raise ValueError("expected an issue, got a pull request")
        return _positive(issue.get("number"), "issue.number")

    def family(self, number: int) -> dict[int, Issue]:
        pending = deque([_positive(number, "issue")])
        issues: dict[int, Issue] = {}
        while pending:
            current = pending.popleft()
            if current in issues:
                continue
            base = f"repos/{self.repo}/issues/{current}"
            raw = _object(self.get(base), "issue")
            require_work_issue(raw, allow_proposal=True)
            if self.identity(raw) != current:
                raise ValueError("issue response number differs from request")
            # Only this endpoint's documented 404 means no parent. All other
            # request failures abort the check, without a successful partial report.
            parent_raw = self.get(f"{base}/parent", missing_parent=True)
            parent = self.identity(parent_raw) if parent_raw is not None else None
            children: list[int] = []
            page = 1
            while True:
                rows = self.get(f"{base}/sub_issues?per_page=100&page={page}")
                if not isinstance(rows, list) or len(rows) > 100:
                    raise ValueError(
                        "sub_issues response must be a page of at most 100 issues"
                    )
                for row in rows:
                    require_work_issue(_object(row, "sub-issue"), allow_proposal=True)
                page_children = [self.identity(row) for row in rows]
                if len(set(page_children)) != len(page_children) or set(
                    children
                ).intersection(page_children):
                    raise ValueError("duplicate sub-issue within/across pages")
                children.extend(page_children)
                if len(rows) < 100:
                    break
                page += 1
            labels_raw = raw.get("labels")
            if not isinstance(labels_raw, list):
                raise ValueError("issue.labels must be a list")
            labels = frozenset(
                _string(_object(label, "label").get("name"), "label.name")
                for label in labels_raw
            )
            body_raw = raw.get("body")
            reason_raw = raw.get("state_reason")
            issues[current] = Issue(
                current,
                _string(raw.get("title"), "issue.title"),
                "" if body_raw is None else _string(body_raw, "issue.body"),
                labels,
                parent,
                tuple(children),
                _string(raw.get("state"), "issue.state"),
                None
                if reason_raw is None
                else _string(reason_raw, "issue.state_reason"),
            )
            pending.extend(children)
            if parent is not None:
                pending.append(parent)
        return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="read the entire related issue family and report inconsistencies"
    )
    check.add_argument("--issue", type=int, required=True)
    check.add_argument("--repo", help="owner/repo (default: unique GitHub remote)")
    render = commands.add_parser(
        "render", help="print a local draft as JSON; never write to GitHub"
    )
    render.add_argument("--spec", required=True, help="JSON spec path, or - for stdin")
    render.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="always enabled; this CLI has no write operation",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            content = (
                sys.stdin.read()
                if args.spec == "-"
                else Path(args.spec).read_text(encoding="utf-8")
            )
            result = render_spec(json.loads(content))
            code = 0
        else:
            issues = GitHub(args.repo).family(args.issue)
            problems = check_consistency(issues)
            result = {
                "ok": not problems,
                "issues": sorted(issues),
                "problems": [asdict(problem) for problem in problems],
            }
            code = int(bool(problems))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return code
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"issue-format: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
