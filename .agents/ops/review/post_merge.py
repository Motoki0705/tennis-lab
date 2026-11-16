"""Explicit, numbered confirmation of artifacts merged before pipeline adoption."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from parent.execution import authorization
from parent.policy import closing_issues, object_value
from parent.session import ParentSession

from review.progression import new_state, read, save


def register(session: ParentSession, numbers: list[int]) -> dict[str, Any]:
    from review.merge_gate import state_path

    if not numbers or len(numbers) != len(set(numbers)):
        raise ValueError("confirmation requires unique merged PR numbers")
    session.gh.targeted_history = True
    row = session.agreed()
    _, items = session.work_items()
    auth = authorization(session)
    targets = []
    for n in numbers:
        pr = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
        issues = closing_issues(pr.get("body") or "", session.gh.repo)
        commit = pr.get("merge_commit_sha")
        if (
            pr.get("number") != n
            or pr.get("merged") is not True
            or pr.get("state") != "closed"
            or not pr.get("merged_at")
            or not issues
            or not issues <= set(items)
            or not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or pr["base"]["ref"] != pr["base"]["repo"].get("default_branch")
            or any(
                pr[side]["repo"]["full_name"].casefold() != session.gh.repo.casefold()
                for side in ("head", "base")
            )
        ):
            raise ValueError(
                f"PR #{n}: related default-branch merged artifact is required"
            )
        actual = object_value(
            session.gh.get(f"repos/{session.gh.repo}/commits/{commit}")
        )
        if actual.get("sha") != commit:
            raise ValueError(f"PR #{n}: merge commit is unavailable")
        targets.append(
            {
                "number": n,
                "issues": sorted(issues),
                "dependencies": [],
                "worker": None,
                "head_sha": pr["head"]["sha"],
                "branch": pr["head"]["ref"],
                "base_ref": pr["base"]["ref"],
                "merge_head": pr["head"]["sha"],
                "merge_commit": commit,
            }
        )
    for identifier in row.get("post_merge_confirmations", []):
        state = read(state_path(session.store, identifier))
        if set(state["targets"]) == {str(n) for n in numbers}:
            if any(
                state["targets"][str(t["number"])]["merge_commit"] != t["merge_commit"]
                for t in targets
            ):
                raise ValueError("previously confirmed merge commit changed")
            return {
                "review_run_id": identifier,
                "reused": True,
                "status": state["status"],
            }
    identifier = f"parent-{session.parent}-{uuid.uuid4().hex}"
    request = {
        "schema_version": 4,
        "kind": "post_merge_confirmation",
        "review_run_id": identifier,
        "repo": session.gh.repo,
        "parent_issue": session.parent,
        "authorization": auth,
        "prs": targets,
    }
    state = new_state(request, owner=row["session"], generation=row["generation"])
    state.update(checkout=str(Path.cwd()), options={})
    state["review"] = {"stage": "not_required", "round": 0}
    for target in state["targets"].values():
        # The wait deadline starts when confirmation is first attempted, not at
        # the historical merge date. An interrupted run keeps that deadline.
        target.update(stage="post_merge_wait", merged_at=None)
    path = state_path(session.store, identifier)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(path, state)
    row.setdefault("post_merge_confirmations", []).append(identifier)
    row["latest_progression_id"] = identifier
    session.store.save()
    from decision_logs.review_capture import bind_request

    bind_request(identifier)
    return {"review_run_id": identifier, "reused": False, "status": "requested"}
