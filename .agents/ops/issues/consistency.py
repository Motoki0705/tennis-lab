"""Validate a complete connected family of issue snapshots, without I/O."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .format import (
    CHILD_LABEL,
    PARENT_LABEL,
    TASKS_HEADING,
    ParsedTitle,
    TaskStatus,
    parent_reference,
    parse_child_tasks,
    parse_sections,
    parse_title,
    validate_body,
    visible_lines,
)


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    labels: frozenset[str]
    parent: int | None = None
    children: tuple[int, ...] = ()
    state: str = "open"
    state_reason: str | None = None


@dataclass(frozen=True, order=True)
class Problem:
    issue: int
    code: str
    message: str


def task_status(issue: Issue) -> TaskStatus:
    if issue.state == "open":
        return "open"
    if issue.state == "closed":
        if issue.state_reason == "completed":
            return "completed"
        if issue.state_reason in {"not_planned", "duplicate"}:
            return "cancelled"
    raise ValueError(f"unknown issue state/reason: {issue.state}/{issue.state_reason}")


def check_consistency(issues: Mapping[int, Issue]) -> tuple[Problem, ...]:
    """Relations are authoritative; labels alone never imply a relationship.

    Include the entire connected family, including closed/cancelled children.
    Missing snapshots, contradictory edges and cycles are errors.
    """
    problems: list[Problem] = []
    titles: dict[int, ParsedTitle] = {}
    for number, issue in sorted(issues.items()):

        def add(code: str, message: str, n: int = number) -> None:
            problems.append(Problem(n, code, message))

        if type(number) is not int or number < 1 or issue.number != number:
            add("snapshot", "mapping key must equal a positive issue number")
        try:
            task_status(issue)
        except ValueError as exc:
            add("state", str(exc))
        # These are hints that require agreement with actual relations, never a
        # substitute for a missing GitHub edge. Losing both labels must not hide
        # a stale child prefix/reference or a parent's checklist.
        try:
            title_hint = parse_title(issue.title)
        except ValueError:
            title_hint = None
        try:
            has_checklist = TASKS_HEADING in parse_sections(issue.body)
            has_reference = any(
                line.startswith("親issue:") for line in visible_lines(issue.body)
            )
        except ValueError:
            has_checklist = has_reference = False
        is_parent = (
            PARENT_LABEL in issue.labels or bool(issue.children) or has_checklist
        )
        is_child = (
            CHILD_LABEL in issue.labels
            or issue.parent is not None
            or bool(title_hint and title_hint.parent)
            or has_reference
        )
        if is_parent and PARENT_LABEL not in issue.labels:
            add("label.parent", f"missing {PARENT_LABEL}")
        if is_child and CHILD_LABEL not in issue.labels:
            add("label.child", f"missing {CHILD_LABEL}")
        if is_child and issue.parent is None:
            add("relationship.parent", "child has no GitHub parent")
        if is_parent or is_child:
            try:
                title = titles[number] = parse_title(issue.title)
                if (title.identifier is not None) != is_parent:
                    add(
                        "title.identifier",
                        "own identifier must be present only for a parent",
                    )
                if (title.parent.number if title.parent else None) != issue.parent:
                    add("title.parent", "title parent differs from GitHub parent")
            except ValueError as exc:
                add("title", str(exc))
            for error in validate_body(
                issue.body, is_parent=is_parent, is_child=is_child
            ):
                add("body", error)
            try:
                if parent_reference(issue.body) != issue.parent:
                    add("body.parent", "body parent differs from GitHub parent")
            except ValueError:
                pass  # Already reported by validate_body.
        if issue.parent is not None:
            if issue.parent not in issues:
                add("relationship.parent", f"missing parent snapshot #{issue.parent}")
            elif number not in issues[issue.parent].children:
                add(
                    "relationship.parent",
                    f"parent #{issue.parent} does not list this child",
                )
        if len(set(issue.children)) != len(issue.children):
            add("relationship.children", "duplicate sub-issue")
        for child_number in issue.children:
            if child_number not in issues:
                add("relationship.children", f"missing child snapshot #{child_number}")
            elif issues[child_number].parent != number:
                add(
                    "relationship.children",
                    f"#{child_number} names another GitHub parent",
                )
        if is_parent:
            try:
                tasks = parse_child_tasks(
                    parse_sections(issue.body).get(TASKS_HEADING, "")
                )
            except ValueError:
                continue  # Already reported by validate_body.
            actual = set(issue.children)
            listed = {task.number for task in tasks}
            if actual != listed:
                add(
                    "checklist.members",
                    f"sub-issues/checklist differ: missing={sorted(actual - listed)}, extra={sorted(listed - actual)}",
                )
            for task in tasks:
                child = issues.get(task.number)
                if child is None or task.number not in actual:
                    continue
                try:
                    if task.status != task_status(child):
                        add(
                            "checklist.state",
                            f"#{task.number}: checklist status differs from GitHub state/reason",
                        )
                except ValueError:
                    pass  # Reported on the child's snapshot.
                try:
                    if task.title != parse_title(child.title).summary:
                        add(
                            "checklist.title",
                            f"#{task.number}: checklist task name differs from child summary",
                        )
                except ValueError:
                    pass

    for number, issue in sorted(issues.items()):
        own_title = titles.get(number)
        parent_title = titles.get(issue.parent) if issue.parent is not None else None
        if (
            own_title
            and own_title.parent
            and parent_title
            and own_title.parent.identifier != parent_title.identifier
        ):
            problems.append(
                Problem(
                    number,
                    "title.prefix",
                    "parent identifier differs from the parent's own identifier",
                )
            )
        visited: set[int] = set()
        cursor: int | None = number
        while cursor is not None and cursor in issues:
            if cursor in visited:
                problems.append(
                    Problem(
                        number,
                        "relationship.cycle",
                        "GitHub parent relations contain a cycle",
                    )
                )
                break
            visited.add(cursor)
            cursor = issues[cursor].parent
    return tuple(sorted(set(problems)))
