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
    session: ParentSession,
    selected: list[int],
    reason: str | None,
    *,
    review_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from lib.review_settings import choose, load_policy, validate_selection

    from .review_choice import verify_references

    choice = (
        validate_selection(review_selection)
        if review_selection is not None
        else choose(load_policy())
    )
    row = session.agreed()
    auth = authorization(session)
    parent, children = session.work_items()
    child_numbers = set(children)
    prs = select_prs(selected, session.gh.pulls(), child_numbers, session.gh.repo)
    from .worker_preflight import require_implementation

    for pr in prs:
        for child in (
            closing_issues(pr.get("body") or "", session.gh.repo) & child_numbers
        ):
            require_implementation(session, child)
    entries, diffs = [], {}
    for item in prs:
        entry, diff = snapshot(dict(item))
        entries.append(entry)
        diffs[item["number"]] = diff
    # Protect the collection window: changes during git I/O are not covered.
    with session.gh.fresh_reads():
        if set(session.work_items()[1]) != child_numbers:
            raise ValueError("work targets changed while freezing review targets")
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
    verify_references(session, choice, entries)
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
        and reviews[-1]["request"].get("review_selection") == choice
        and set(reviews[-1]["request"].get("requirements", requirements))
        == set(requirements)
        and (reason is None or reason == reviews[-1]["request"].get("reason"))
    ):
        request = object_value(reviews[-1]["request"])
        require_run_authorization(session, request)
        publish_targets(session, request)
        return request
    previously_reviewed = {p["number"] for r in reviews for p in r["request"]["prs"]}
    if previously_reviewed.intersection(selected) and not reason:
        raise ValueError(
            "a new review run requires an explicit reason; no automatic re-review"
        )
    if reason:
        nonempty(reason, "new review reason")
    run_id = "parent-" + str(session.parent) + "-" + uuid.uuid4().hex
    request = {
        "schema_version": 3,
        "review_selection": choice,
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
    session: ParentSession,
    run_id: str,
    target: str,
    evidence: str,
    *,
    review_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = session.agreed()
    nonempty(evidence, "review instruction/evidence")
    if target not in REVIEW_STATES:
        raise ValueError("unknown review state")
    run = next(
        (item for item in row["reviews"] if item["request"]["review_run_id"] == run_id),
        None,
    )
    if run is None:
        raise ValueError("unknown review run")
    if (
        review_selection is not None
        and run["request"].get("review_selection") != review_selection
    ):
        raise ValueError("review selection is frozen; prepare a new run with reason")
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
    if target != current and not (
        REVIEW_STATES.index(target) == REVIEW_STATES.index(current) + 1
        or (current == "instructed" and target == "fixes_verified")
    ):
        raise ValueError(f"invalid review transition {current} -> {target}")
    if target == "instructed":
        request = run["request"]
        parent, children = session.work_items()
        from .worker_preflight import require_implementation

        for entry in request["prs"]:
            raw = session.gh.get(f"repos/{session.gh.repo}/pulls/{entry['number']}")
            for child in closing_issues(raw.get("body") or "", session.gh.repo) & (
                set(children)
            ):
                require_implementation(session, child)
        prs = select_prs(
            [p["number"] for p in request["prs"]],
            session.gh.pulls(),
            set(children),
            session.gh.repo,
        )
        require_requirement_targets(
            request,
            {
                n
                for pr in prs
                for n in closing_issues(pr.get("body") or "", session.gh.repo)
                & set(children)
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
