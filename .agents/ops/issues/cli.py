"""Read-only gh adapter and local draft renderer. No GitHub mutation commands."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .consistency import Issue, check_consistency
from .format import (
    CHILD_LABEL,
    PARENT_LABEL,
    ChildTask,
    ParentRef,
    ProposalSource,
    TaskStatus,
    child_title,
    parent_title,
    render_child_body,
    render_parent_body,
)


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


class GitHubHTTPError(ValueError):
    """A failed gh api response with its actual HTTP status preserved."""

    def __init__(self, endpoint: str, status: int, detail: str) -> None:
        self.status = status
        super().__init__(f"gh api {endpoint}: HTTP {status} ({detail})")


class GitHub:
    def __init__(self, repo: str | None) -> None:
        if repo is None:
            result = self._run("repo", "view", "--json", "nameWithOwner")
            if result.returncode:
                raise ValueError(f"gh repo view failed: {result.stderr.strip()}")
            repo = _string(
                _object(json.loads(result.stdout), "repository").get("nameWithOwner"),
                "nameWithOwner",
            )
        self.repo = _repo_name(repo)

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["gh", *args], text=True, capture_output=True, timeout=60, check=False
        )

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
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
        headers, separator, body = result.stdout.replace("\r\n", "\n").partition("\n\n")
        status = re.match(r"HTTP/\S+ ([0-9]{3})(?:\s|$)", headers)
        if not separator or not status:
            raise ValueError(
                f"gh api {endpoint}: missing HTTP response status ({result.stderr.strip()})"
            )
        payload: object = json.loads(body)
        if status[1] == "404" and result.returncode != 0 and missing_parent:
            error = _object(payload, "parent 404")
            if error.get("message") == "No parent issue found":
                return None
        if result.returncode or status[1] != "200":
            raise GitHubHTTPError(endpoint, int(status[1]), result.stderr.strip())
        if payload is None:
            raise ValueError(f"gh api {endpoint}: unexpected null response")
        return payload

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
    check.add_argument("--repo", help="owner/repo (default: gh repo view)")
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
