"""Worker identity within an Issue. The existing task record owns the closing PR."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any, cast

PRIMARY = "primary"


def select(
    row: Mapping[str, Any], child: int, name: str | None, *, creating: bool = False
) -> str:
    task = row.get("tasks", {}).get(str(child), {})
    helpers = task.get("helpers", {})
    if name is None:
        if helpers:
            raise ValueError(
                "--worker is required: " + ", ".join([PRIMARY, *sorted(helpers)])
            )
        return PRIMARY
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name):
        raise ValueError("invalid worker name")
    if not creating and name != PRIMARY and name not in helpers:
        raise ValueError(f"unknown worker {name} for issue #{child}")
    return name


def record(row: Mapping[str, Any], child: int, name: str = PRIMARY) -> dict[str, Any]:
    task = row.get("tasks", {}).get(str(child), {})
    if name == PRIMARY:
        return cast(dict[str, Any], task)
    return cast(dict[str, Any], task.get("helpers", {}).get(name, {}))


def workers(tasks: Mapping[str, Any]) -> Iterator[tuple[int, str, dict[str, Any]]]:
    for child, task in tasks.items():
        yield int(child), PRIMARY, task
        for name, helper in task.get("helpers", {}).items():
            yield int(child), name, helper


def job_record(row: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    return record(row, job["child"], job.get("worker_name", PRIMARY))
