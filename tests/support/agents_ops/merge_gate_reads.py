"""Multi-parent fake inputs and request summaries for merge-gate diagnostics."""

from __future__ import annotations

import copy
import importlib
import re
from collections import Counter
from pathlib import Path
from typing import Any


def add_unrelated_parents(p: Any, count: int) -> set[int]:
    """Add real formal families, evidence and disjoint review registrations."""
    state = p.state()
    unrelated: set[int] = set()
    for index in range(count):
        parent = 1000 + index * 10
        children = [parent + 1, parent + 2]
        evidence = parent + 3
        for n in [parent, *children, evidence]:
            state["issues"][str(n)] = {
                **copy.deepcopy(state["issues"]["1"]),
                "id": 10000 + n,
                "number": n,
                "title": f"Unrelated issue {n}",
                "user": {"login": "operator"},
            }
            state["comments"][str(n)] = []
        state["issues"][str(evidence)].update(
            body=importlib.import_module("parent.evidence").issue_body("o/r", parent),
            labels=[{"name": "parent-evidence"}],
        )
        for child in children:
            state["parents"][str(child)] = parent
            state["issues"][str(child)]["labels"] = [{"name": "issue:child"}]
            state["prs"].append(
                {
                    **copy.deepcopy(state["prs"][0]),
                    "number": child + 10000,
                    "body": f"Closes #{child}",
                }
            )
        unrelated.update([parent, *children, evidence])
    p.save(state)
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    store = importlib.import_module("parent.state").Store(directory)
    with store.locked():
        template = store.parents["o/r#1"]
        for index in range(count):
            parent = 1000 + index * 10
            row = copy.deepcopy(template)
            row.update(session=f"claude:unrelated-parent-{parent}", tasks={})
            for review in row["reviews"]:
                request = review["request"]
                request["parent_issue"] = parent
                request["review_run_id"] = f"parent-{parent}-" + f"{parent:032x}"
                request["prs"] = [
                    {**pr, "number": parent + 10001 + i}
                    for i, pr in enumerate(request["prs"][:2])
                ]
            store.parents[f"o/r#{parent}"] = row
        store.save()
    return unrelated


def api_endpoints(calls: list[list[str]]) -> list[str]:
    return [
        next(a for a in call if a in {"user", "graphql"} or a.startswith("repos/"))
        for call in calls
        if call[0] == "api"
    ]


def endpoint_counts(endpoints: list[str]) -> dict[str, int]:
    """Group IDs only for reporting; exact endpoint assertions use raw calls."""
    return dict(
        sorted(
            Counter(
                re.sub(
                    r"/\d+", "/N", re.sub(r"/commits/[0-9a-f]{40}", "/commits/HEAD", e)
                )
                for e in endpoints
            ).items()
        )
    )
