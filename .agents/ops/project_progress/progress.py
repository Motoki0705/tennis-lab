"""Progress denominators use formal relationships and issue reasons, not checkboxes."""

from __future__ import annotations

from collections import deque
from typing import Any

from issues import PARENT_LABEL, TASKS_HEADING, parse_child_tasks, parse_sections
from parent.github import GitHub
from parent.policy import closing_issues, nonempty, number, object_value

from .github import Projects


def cancelled(issue: dict[str, Any]) -> bool:
    return issue["state"] == "CLOSED" and issue["stateReason"] in {
        "NOT_PLANNED",
        "DUPLICATE",
    }


def completed(issue: dict[str, Any]) -> bool:
    return bool(issue["state"] == "CLOSED" and issue["stateReason"] == "COMPLETED")


def parent_issue(issue: dict[str, Any], root: int) -> bool:
    return bool(
        issue["number"] == root or issue["children"] or PARENT_LABEL in issue["labels"]
    )


def read_tree(api: Projects, repo: str, root: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    expected_parent: dict[int, int | None] = {root: None}
    pending = deque([root])
    while pending:
        current = pending.popleft()
        raw = api.issue(repo, current)
        valid = (
            raw["state"] == "OPEN" and raw["stateReason"] in {None, "REOPENED"}
        ) or (
            raw["state"] == "CLOSED"
            and raw["stateReason"] in {"COMPLETED", "NOT_PLANNED", "DUPLICATE"}
        )
        if not valid:
            raise ValueError(f"#{current}: unknown issue state/reason")
        parent = raw["parent"]
        expected = expected_parent[current]
        if expected is not None and (
            parent is None
            or parent["number"] != expected
            or parent["repository"]["nameWithOwner"].casefold() != repo.casefold()
            or parent["id"] != result[expected]["id"]
        ):
            raise ValueError(f"#{current}: formal parent changed/inconsistent")
        summary = object_value(raw["subIssuesSummary"])
        if (
            any(
                type(summary[key]) is not int or summary[key] < 0
                for key in ("total", "completed", "percentCompleted")
            )
            or summary["completed"] > summary["total"]
            or summary["percentCompleted"] > 100
        ):
            raise ValueError(f"#{current}: invalid standard sub-issue summary")
        nonempty(raw["title"], "issue title")
        nonempty(raw["url"], "issue URL")
        result[current] = raw
        for child in raw["children"]:
            if child in expected_parent:
                raise ValueError("cycle/repeated issue in formal hierarchy")
            expected_parent[child] = current
            pending.append(child)
    return result


def related_pulls(
    gh: GitHub, tree: dict[int, dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {n: [] for n in tree}
    seen = set()
    for raw in gh.pulls(set(tree)):
        n = number(raw["number"])
        if n in seen:
            raise ValueError("duplicate PR within/across pages")
        seen.add(n)
        targets = closing_issues(raw.get("body") or "", gh.repo) & tree.keys()
        if not targets:
            continue
        if "merged_at" not in raw or raw["state"] not in {"open", "closed"}:
            raise ValueError(f"PR #{n}: missing merge/state information")
        merged = raw["merged_at"] is not None
        if merged and (
            raw["state"] != "closed" or not isinstance(raw["merged_at"], str)
        ):
            raise ValueError(f"PR #{n}: inconsistent merge/state information")
        row = {
            "number": n,
            "url": nonempty(raw["html_url"], "PR URL"),
            "state": "merged"
            if merged
            else "open"
            if raw["state"] == "open"
            else "closed_unmerged",
        }
        for target in targets:
            result[target].append(row)
    return {
        n: sorted(rows, key=lambda row: row["number"]) for n, rows in result.items()
    }


def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = [row["number"] for row in rows if cancelled(row)]
    return {
        "completed": sum(completed(row) for row in rows if not cancelled(row)),
        "total": sum(not cancelled(row) for row in rows),
        "cancelled": excluded,
    }


def progress(tree: dict[int, dict[str, Any]], n: int) -> dict[str, Any]:
    issue = tree[n]
    direct = [tree[child] for child in issue["children"]]
    leaves = []
    excluded_subtrees = []
    pending = deque(direct)
    while pending:
        row = pending.popleft()
        if cancelled(row):
            excluded_subtrees.append(row["number"])
        elif row["children"]:
            pending.extend(tree[child] for child in row["children"])
        else:
            leaves.append(row)
    section = parse_sections(issue["body"] or "").get(TASKS_HEADING)
    checklist: dict[str, Any] = {"available": section is not None}
    if section is not None:
        try:
            tasks = parse_child_tasks(section)
        except ValueError as exc:
            checklist["error"] = str(exc)
        else:
            by_number = {task.number: task for task in tasks}
            checklist.update(
                total=len(tasks),
                checked=sum(task.status == "completed" for task in tasks),
                missing=sorted(set(issue["children"]) - by_number.keys()),
                not_direct=sorted(by_number.keys() - set(issue["children"])),
                completion_disagreements=[
                    row["number"]
                    for row in direct
                    if row["number"] in by_number
                    and (by_number[row["number"]].status == "completed")
                    != completed(row)
                ],
            )
    return {
        "direct": tally(direct),
        "leaves": {**tally(leaves), "excluded_subtrees": excluded_subtrees},
        "standard": issue["subIssuesSummary"],
        "checklist": checklist,
    }


def progress_text(value: dict[str, Any]) -> str:
    direct, leaves, standard, checklist = (
        value[key] for key in ("direct", "leaves", "standard", "checklist")
    )
    text = (
        f"直属 {direct['completed']}/{direct['total']}（取消{len(direct['cancelled'])}除外）; "
        f"末端 {leaves['completed']}/{leaves['total']}（入れ子を重複計上しない）; "
        f"標準 {standard['completed']}/{standard['total']}"
    )
    if "checked" in checklist:
        text += f"; 本文チェック {checklist['checked']}/{checklist['total']}（実施判定に不使用）"
        if any(
            checklist[key]
            for key in ("missing", "not_direct", "completion_disagreements")
        ):
            text += "・正式関係/完了チェックに差異あり"
    elif "error" in checklist:
        text += "; 本文チェックは書式エラー（inspectで確認）"
    else:
        text += "; 本文チェックリストなし"
    return text
