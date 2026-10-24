"""Git snapshots and the explicit review handoff. Never launches a reviewer."""

from __future__ import annotations

import hashlib
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .execution import (
    authorization,
    require_requirement_targets,
    require_run_authorization,
    requirement_body,
    task_errors,
)
from .policy import (
    REVIEW_LABELS,
    REVIEW_STATES,
    closing_issues,
    digest,
    nonempty,
    object_value,
    review_state,
    select_prs,
)
from .session import ParentSession


def git(*args: str, cwd: Path | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, timeout=60, check=False
    )
    if result.returncode:
        raise ValueError(
            f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def validate_worktree(worktree: Path, branch: str) -> None:
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip()
    )
    main_root = common.parent
    resolved = worktree.resolve(strict=True)
    expected = main_root / ".claude/worktrees"
    if not resolved.is_relative_to(expected) or resolved in (expected, main_root):
        raise ValueError(
            "worker worktree must be under the main checkout's .claude/worktrees/"
        )
    if (
        Path(
            git("rev-parse", "--show-toplevel", cwd=resolved).decode().strip()
        ).resolve()
        != resolved
    ):
        raise ValueError("worker path must be a worktree root")
    worker_common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=resolved)
        .decode()
        .strip()
    )
    if (
        worker_common != common
        or git("branch", "--show-current", cwd=resolved).decode().strip() != branch
    ):
        raise ValueError("worker must belong to this repository and requested branch")
    if resolved == root.resolve():
        raise ValueError("a child requires its own worktree, separate from the parent")


def snapshot(pr: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    if any(
        not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha)
        for sha in (head, base)
    ):
        raise ValueError("PR head/base must be full git SHA-1 identifiers")
    for sha in (head, base):
        available = subprocess.run(
            ["git", "cat-file", "-e", sha + "^{commit}"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if available.returncode:
            git("fetch", "--no-tags", "--no-write-fetch-head", "origin", sha)
    merge_base = git("merge-base", base, head).decode().strip()
    diff = git(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-color",
        merge_base,
        head,
        "--",
    )
    return {
        "number": pr["number"],
        "head_sha": head,
        "base_sha": base,
        "base_ref": pr["base"]["ref"],
        "merge_base_sha": merge_base,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "branch": pr["head"]["ref"],
    }, diff


def prepare(
    session: ParentSession, selected: list[int], reason: str | None
) -> dict[str, Any]:
    row = session.agreed()
    auth = authorization(session)
    parent, children = session.family()
    child_numbers = set(children) or {parent.number}
    prs = select_prs(selected, session.gh.pulls(), child_numbers, session.gh.repo)
    for pr in prs:
        for child in closing_issues(pr.get("body") or "", session.gh.repo) & set(
            children
        ):
            errors = task_errors(
                row["tasks"].get(str(child), {}), auth, children[child].body
            )
            if errors:
                raise ValueError(
                    f"#{child}: stale delegation/result: {errors}; verify it with record-result before preparing review"
                )
    entries, diffs = [], {}
    for item in prs:
        entry, diff = snapshot(dict(item))
        entries.append(entry)
        diffs[item["number"]] = diff
    # Protect the collection window: changes during git I/O are not covered.
    with session.gh.fresh_reads():
        fresh = select_prs(selected, session.gh.pulls(), child_numbers, session.gh.repo)
    for before, after in zip(prs, fresh, strict=True):
        if any(
            before[part][field] != after[part][field]
            for part, field in (
                ("head", "sha"),
                ("base", "sha"),
                ("base", "ref"),
                ("head", "ref"),
            )
        ) or (
            closing_issues(before.get("body") or "", session.gh.repo) & child_numbers
            != closing_issues(after.get("body") or "", session.gh.repo) & child_numbers
        ):
            raise ValueError(
                "PR changed while freezing review targets; retry explicitly"
            )
    requirements = {str(parent.number): digest(requirement_body(parent.body))}
    for pr in prs:
        for n in closing_issues(pr.get("body") or "", session.gh.repo) & set(children):
            requirements[str(n)] = digest(requirement_body(children[n].body))
    reviews = row["reviews"]
    if reviews:
        public_state = review_state(session.history(), reviews[-1]["request"])
        if public_state:
            reviews[-1]["state"] = public_state
    if (
        reviews
        and reviews[-1]["state"] == "targets_fixed"
        and reviews[-1]["request"]["prs"] == entries
        and (
            (
                reviews[-1]["request"].get("authorization") == auth
                and reviews[-1]["request"].get("requirements") == requirements
            )
            or reviews[-1]["request"]["schema_version"] == 1
        )
    ):
        request = object_value(reviews[-1]["request"])
        require_run_authorization(session, request)
        publish_targets(session, request)
        return request
    if reviews and not reason:
        raise ValueError(
            "a new review run requires an explicit reason; no automatic re-review"
        )
    if reason:
        nonempty(reason, "new review reason")
    run_id = "parent-" + str(session.parent) + "-" + uuid.uuid4().hex
    request = {
        "schema_version": 2,
        "authorization": auth,
        "requirements": requirements,
        "review_run_id": run_id,
        "repo": session.gh.repo,
        "parent_issue": session.parent,
        "prs": entries,
        "reason": reason or "親が初回の対象PR群を確定",
    }
    session.public(request)
    require_run_authorization(session, request)
    run_dir = session.store.directory / "reviews" / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    for n, diff in diffs.items():
        (run_dir / f"pr-{n}.diff").write_bytes(diff)
    session.store.write_json(run_dir / "request.json", request)
    reviews.append({"state": "targets_fixed", "request": request})
    session.store.save()
    publish_targets(session, request)
    return request


def publish_targets(session: ParentSession, request: dict[str, Any]) -> None:
    import json

    session.post(
        {
            "event": "review",
            "state": "targets_fixed",
            "review_run_id": request["review_run_id"],
            "request_sha256": digest(request),
            **require_run_authorization(session, request),
        },
        "レビュー対象確定。実行は未指示。\n\n```json\n"
        + json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```",
    )


def transition(
    session: ParentSession, run_id: str, target: str, evidence: str
) -> dict[str, Any]:
    row = session.agreed()
    nonempty(evidence, "review instruction/evidence")
    if target not in REVIEW_STATES:
        raise ValueError("unknown review state")
    if not row["reviews"] or row["reviews"][-1]["request"]["review_run_id"] != run_id:
        raise ValueError("unknown or superseded review run")
    run = row["reviews"][-1]
    # A run's progress does not grant coverage to a changed child. Gate checks
    # each PR separately so a stopped child does not stop independent work.
    auth = require_run_authorization(
        session,
        run["request"],
        required_issues=None if target == "instructed" else set(),
    )
    current = review_state(session.history(), run["request"])
    if current is None:
        raise ValueError("review target record is missing from the parent issue")
    if target == current and target != "instructed":
        run["state"] = current
        session.store.save()
        return {**run["request"], "review_state": target}
    if (
        target != current
        and REVIEW_STATES.index(target) != REVIEW_STATES.index(current) + 1
    ):
        raise ValueError(f"invalid review transition {current} -> {target}")
    if target == "instructed":
        request = run["request"]
        parent, children = session.family()
        prs = select_prs(
            [p["number"] for p in request["prs"]],
            session.gh.pulls(),
            set(children) or {parent.number},
            session.gh.repo,
        )
        require_requirement_targets(
            request,
            {
                n
                for pr in prs
                for n in closing_issues(pr.get("body") or "", session.gh.repo)
                & (set(children) or {parent.number})
            },
            exact=True,
        )
        entries = [snapshot(dict(pr))[0] for pr in prs]
        if entries != request["prs"]:
            raise ValueError(
                "review target changed since snapshot; prepare a new run with reason"
            )
        # Snapshot must also have been durably recorded publicly before instruction.
        if not any(
            e.get("event") == "review"
            and e.get("state") == "targets_fixed"
            and e.get("review_run_id") == run_id
            and e.get("request_sha256") == digest(request)
            for e in session.history()
        ):
            raise ValueError("review target record is missing from the parent issue")
    if target == current:
        run["state"] = current
        session.store.save()
        return {**run["request"], "review_state": target}
    session.post(
        {
            "event": "review",
            "state": target,
            "review_run_id": run_id,
            "request_sha256": digest(run["request"]),
            **auth,
        },
        f"レビュー状態: {REVIEW_LABELS[target]}\n\n{evidence}",
    )
    run["state"] = target
    run["evidence"] = evidence
    session.store.save()
    # This is the #1218 input, not an execution. Same run ID must be consumed at most once there.
    return {**run["request"], "review_state": target}
