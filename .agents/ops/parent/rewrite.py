"""Wait for the issue rewrite workflow and gate requirement recording."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from issues.cli import GitHubHTTPError
from issues.rewrite_state import (
    REWRITE_SKIPPED_LABEL,
    RewriteState,
    read_enabled_since,
    rewrite_state,
)

from .github import GitHub
from .policy import nonempty, number, object_value

WORKFLOW = "issue-rewrite.yml"


def current_state(gh: GitHub, parent: int) -> RewriteState:
    with gh.fresh_reads():
        enabled_since = read_enabled_since(gh)
        raw = gh.issue(parent)
    return rewrite_state(
        [label["name"] for label in raw["labels"]],
        raw["created_at"],
        enabled_since,
    )


def require_final(gh: GitHub, parent: int) -> None:
    if current_state(gh, parent) == "pending":
        raise ValueError(
            f"#{parent}: issue rewrite is pending; run parent --parent {parent} "
            f"wait-rewrite, or ask a human to add {REWRITE_SKIPPED_LABEL} to proceed"
        )


def latest_run(gh: GitHub, parent: int) -> int | None:
    page = 1
    seen: set[int] = set()
    while True:  # Paginate once; do not poll for a run to appear.
        try:
            raw = object_value(
                gh.get(
                    f"repos/{gh.repo}/actions/workflows/{WORKFLOW}/runs"
                    f"?per_page=100&page={page}"
                )
            )
        except GitHubHTTPError as exc:
            if exc.status == 404 and page == 1:
                return None
            raise
        runs = raw.get("workflow_runs")
        if not isinstance(runs, list) or len(runs) > 100:
            raise ValueError("expected a workflow_runs page of at most 100 runs")
        matching: list[tuple[str, int]] = []
        for value in runs:
            run = object_value(value)
            run_id = number(run.get("id"))
            if run_id in seen:
                raise ValueError("duplicate workflow run within/across pages")
            seen.add(run_id)
            created_at = nonempty(run.get("created_at"), "run created_at")
            if run.get("display_title") == f"issue-rewrite #{parent}":
                matching.append((created_at, run_id))
        if matching:
            return max(matching)[1]
        if len(runs) < 100:
            return None
        page += 1


def wait_rewrite(gh: GitHub, parent: int) -> dict[str, Any]:
    state = current_state(gh, parent)
    if state != "pending":
        return {"parent": parent, "rewrite_state": state, "run_id": None}
    run_id = latest_run(gh, parent)
    if run_id is None:
        # The workflow may have finished while its runs were being fetched.
        state = current_state(gh, parent)
        if state != "pending":
            return {"parent": parent, "rewrite_state": state, "run_id": None}
        raise ValueError(
            f"#{parent}: no matching {WORKFLOW} run; manually start workflow_dispatch "
            f"for this issue, or ask a human to add {REWRITE_SKIPPED_LABEL}"
        )
    for followup in range(2):
        result = subprocess.run(
            ["gh", "run", "watch", str(run_id), "--repo", gh.repo],
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"gh run watch {run_id} failed: exit {result.returncode}")
        state = current_state(gh, parent)
        if state != "pending" or followup:
            break
        # attempt=2 is a distinct workflow_dispatch run. Follow it once, without
        # polling or treating an older/deleted run as a new attempt.
        with gh.fresh_reads():
            newer = latest_run(gh, parent)
        if newer is None or newer <= run_id:
            break
        run_id = newer
    if state == "pending":
        raise ValueError(
            f"#{parent}: run {run_id} ended but issue rewrite is still pending; "
            f"inspect the run, retry workflow_dispatch or ask a human to add {REWRITE_SKIPPED_LABEL}"
        )
    return {"parent": parent, "rewrite_state": state, "run_id": run_id}
