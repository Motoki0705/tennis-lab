"""Explicit GET-only parent snapshots and the existing parent launch plan."""

from __future__ import annotations

from typing import Any

from parent.github import GitHub
from parent.policy import child_stopped, closing_issues, raw_section
from parent.session import ParentSession, task_title
from parent.state import Store

from .safety import Guard, StatusError
from .task_view import remote_title


class ReadOnlyGitHub(GitHub):
    """Cache this observation only; never bind to a locking Store transaction."""

    def __init__(self, repo: str) -> None:
        super().__init__(repo)
        self.snapshot: dict[tuple[str, bool], object] = {}

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        key = endpoint, missing_parent
        if key not in self.snapshot:
            self.snapshot[key] = super().get(endpoint, missing_parent=missing_parent)
        return self.snapshot[key]

    def write(
        self, method: str, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise StatusError("github_write_forbidden")


def stop_detail(body: str, guard: Guard) -> dict[str, Any]:
    import re

    section = raw_section(body, "作業状況・停止事項")
    records = list(
        re.finditer(
            r"(?m)^<!-- parent-task-status:v1 state=(?:stopped|resumed) detail=[0-9a-f]{64} -->$",
            section,
        )
    )
    if not records:
        return {}
    start = records[-2].end() if len(records) > 1 else 0
    latest = section[start : records[-1].start()]
    result = {}
    for key in ("reason", "facts", "impact", "resume_condition", "next_action"):
        match = re.search(
            rf"(?ms)^{key}: (.*?)(?=\n\n(?:reason|facts|impact|resume_condition|next_action): |\Z)",
            latest,
        )
        result[key] = guard.text(match[1].strip() if match else None)
    return result


def parent_snapshot(
    repo: str, parent: int, store: Store, guard: Guard
) -> dict[int, dict[str, Any]]:
    gh = ReadOnlyGitHub(repo)
    # Same snapshot/family and launch_plan path as parent status, without
    # calling status (which also returns private tasks/worktrees/review data).
    session = ParentSession(gh, store, parent, None)
    _, children = session.work_items()
    plan = session.launch_plan()
    pulls = gh.pulls()
    tasks = store.parents[f"{repo.casefold()}#{parent}"]["tasks"]
    result: dict[int, dict[str, Any]] = {}
    for child, issue in children.items():
        planned = next(t for t in plan["tasks"] if t["child"] == child)
        stopped = child_stopped(
            issue.body, tasks.get(str(child), {}).get("stopped", False)
        )
        prs = []
        for pr in pulls:
            if child not in closing_issues(pr.get("body") or "", repo):
                continue
            if (
                type(pr.get("number")) is not int
                or pr["number"] < 1
                or pr.get("state") not in {"open", "closed"}
            ):
                raise ValueError("invalid PR state")
            url = pr.get("html_url")
            if (
                not isinstance(url, str)
                or url.casefold()
                != f"https://github.com/{repo}/pull/{pr['number']}".casefold()
            ):
                raise ValueError("unexpected PR URL")
            prs.append(
                {
                    "number": pr["number"],
                    "state": pr["state"],
                    "merged": pr.get("merged_at") is not None,
                    "url": url,
                }
            )
        if issue.state not in {"open", "closed"} or issue.state_reason not in {
            None,
            "completed",
            "not_planned",
            "reopened",
        }:
            raise ValueError("invalid issue state")
        # Codes are structural; messages and stop notes are free text and may
        # be hidden. Never publish the plan's branch or head references.
        reasons = [
            {"code": r["code"], "message": guard.text(r["message"])}
            for r in planned["reasons"]
        ]
        result[child] = {
            "status": "observed",
            "issue": {
                "state": issue.state,
                "state_reason": issue.state_reason,
                "title": remote_title(task_title(issue), guard),
            },
            "pull_requests": prs,
            "launch": {
                "action": planned["action"],
                "state": planned["state"],
                "dependencies": planned["dependencies"],
                "reasons": reasons,
            },
            "stopped": stopped,
            "stop_record": stop_detail(issue.body, guard) if stopped else {},
        }
    return result
