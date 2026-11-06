"""Explicit adoption of legacy results; never manufacture past worker turns."""

from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .execution import (
    observed_review_authorization,
    require_review_history,
    require_review_inputs,
)
from .policy import closing_issues, digest, review_state

if TYPE_CHECKING:
    from .session import ParentSession


def validate_result(
    task: dict[str, Any],
    current: dict[str, Any],
    record: dict[str, Any],
    pr: dict[str, Any],
    reviewed: dict[str, Any],
) -> None:
    verification = task.get("verification", {})
    if (
        record.get("result") != "completed"
        or record.get("cleanup_complete") is not True
        or record.get("exit_code") != 0
        or verification.get("authorization") != current["authorization"]
        or verification.get("requirements_sha256") != current["requirements"]
        or verification.get("preflight_invalidated")
        or not isinstance(verification.get("evidence"), str)
        or not verification["evidence"].strip()
        or pr["number"] not in verification.get("prs", [])
        or current["child"] not in closing_issues(pr.get("body") or "", current["repo"])
        or any(
            pr[side]["repo"]["full_name"].casefold() != current["repo"]
            for side in ("head", "base")
        )
        or pr["head"]["ref"] != current["branch"]
        or any(
            reviewed.get(k) != v
            for k, v in {
                "number": pr["number"],
                "head_sha": pr["head"]["sha"],
                "branch": pr["head"]["ref"],
                "base_ref": pr["base"]["ref"],
            }.items()
        )
        or pr["state"] not in {"open", "closed"}
        or (
            pr.get("merged_at")
            and (pr["state"] != "closed" or pr.get("merged") is not True)
        )
        or (
            pr["state"] == "closed"
            and (pr.get("merged") is not True or not pr.get("merged_at"))
        )
    ):
        raise ValueError(
            "legacy result, termination, verification or reviewed PR does not match"
        )


def saved_review(session: ParentSession, child: int, request: dict[str, Any]) -> None:
    """Reuse the normal frozen-input/history rules without assuming a phase."""
    from .development import load

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
    policy = importlib.import_module("parent_review_policy")
    policy.request_manifest(request, instructed=False)
    if (
        request["repo"].casefold() != session.gh.repo.casefold()
        or request["parent_issue"] != session.parent
    ):
        raise ValueError("legacy review belongs to another parent/repository")
    directory = session.store.directory / "reviews" / request["review_run_id"]
    if load(directory / "request.json") != request:
        raise ValueError("legacy review request differs from the saved request")
    auth = observed_review_authorization(session, request)
    require_review_inputs(session, request, auth, required_issues={child})
    require_review_history(session, request, auth)
    if review_state(session.history(), request) is None:
        raise ValueError("legacy review has no matching public target confirmation")


def merged_evidence(
    session: ParentSession, child: int, pr: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    from .development import load

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
    gate = importlib.import_module("merge_gate")
    auth = observed_review_authorization(session, request)
    if gate.completion(session, pr, write=False) or gate.confirmed_state(
        session.history(), request, auth
    ) not in {"fixes_verified", "complete"}:
        raise ValueError("legacy merge/close or confirmed review is missing")
    journal = load(gate.attempt_path(session, pr["number"], pr["head"]["sha"]))
    snapshot = journal["snapshot"]
    if (
        journal["pr"] != pr["number"]
        or journal["head_sha"] != pr["head"]["sha"]
        or snapshot["head_sha"] != pr["head"]["sha"]
        or snapshot["branch"] != pr["head"]["ref"]
        or snapshot["base_ref"] != pr["base"]["ref"]
        or journal["review_run_id"] != request["review_run_id"]
        or journal["status"] not in {"merged", "attempting", "unconfirmed"}
    ):
        raise ValueError("legacy merge differs from the checked gate journal")
    directory = session.store.directory / "reviews" / request["review_run_id"]
    result = gate.validate_result(load(directory / "result.json"))
    execution = load(directory / "execution.json")
    if (
        result["request"] != request
        or execution.get("session") != session.row["session"]
    ):
        raise ValueError("legacy merged review/result owner does not match")
    errors = gate.publication_errors(
        result,
        execution,
        session.gh.comments(session.parent),
        {
            entry["number"]: session.gh.comments(entry["number"])
            for entry in request["prs"]
        },
        session.login,
        auth,
    )
    if errors:
        raise ValueError(f"legacy review publication is incomplete: {errors}")
    gate.validate_snapshots([snapshot])
    return cast(dict[str, Any], snapshot)


def resolved_snapshot(
    session: ParentSession, request: dict[str, Any], current: dict[str, Any]
) -> None:
    """Adopt an updated head only through the existing published fix evidence."""
    from .development import load

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
    gate = importlib.import_module("merge_gate")
    policy = importlib.import_module("parent_review_policy")
    workspace = importlib.import_module("review_workspace")
    directory = session.store.directory / "reviews" / request["review_run_id"]
    result = policy.validate_result(load(directory / "result.json"))
    execution = load(directory / "execution.json")
    auth = observed_review_authorization(session, request)
    parent_comments = session.gh.comments(session.parent)
    pr_comments = {
        p["number"]: session.gh.comments(p["number"]) for p in request["prs"]
    }
    if (
        result["request"] != request
        or execution.get("session") != session.row["session"]
        or gate.publication_errors(
            result, execution, parent_comments, pr_comments, session.login, auth
        )
        or gate.confirmed_state(session.history(), request, auth)
        not in {"fixes_verified", "complete"}
    ):
        raise ValueError(
            "legacy updated head requires a published and confirmed review"
        )
    n = current["number"]
    resolutions = []
    for record in policy.records(parent_comments, session.login):
        if (
            record["kind"] != "resolution"
            or record["review_run_id"] != request["review_run_id"]
            or record["pr"] != n
        ):
            continue
        key = digest(record)
        saved_path, diff_path = (
            directory / f"resolution-{key}.json",
            directory / f"resolution-{key}.diff",
        )
        if not saved_path.exists() or not diff_path.exists():
            continue
        saved = load(saved_path)
        if (
            saved.get("published") is True
            and saved.get("record") == record
            and record in policy.records(pr_comments[n], session.login)
            and hashlib.sha256(diff_path.read_bytes()).hexdigest()
            == record["fix_diff_sha256"]
        ):
            resolutions.append(record)
    original = next(p for p in request["prs"] if p["number"] == n)
    paths = workspace.changed_paths(original["head_sha"], current["head_sha"])
    coverage = policy.evaluate_coverage(
        result,
        [current],
        resolutions,
        {n: paths},
        fix_diff_digests={
            n: hashlib.sha256(
                workspace.full_diff(original["head_sha"], current["head_sha"])
            ).hexdigest()
        },
    )
    required = {
        (n, f["id"])
        for f in policy.pr_findings(result, n)
        if f["severity"] in {"blocker", "major"}
    }
    if not coverage.covered or required.intersection(coverage.unresolved):
        raise ValueError("legacy updated head has no matching published fix coverage")


def adopt(
    session: ParentSession,
    child: int,
    task: dict[str, Any],
    value: dict[str, Any],
    directory: Path,
    record: dict[str, Any],
    prs: list[dict[str, Any]],
    evidence: str,
) -> None:
    from .snapshot import git, snapshot, validate_worktree

    if len(prs) != 1 or task.get("stopped"):
        raise ValueError(
            "legacy result requires one related PR and an unstopped delegation; use --replan for an open issue"
        )
    pr = prs[0]
    rows = [
        r
        for r in session.row["reviews"]
        if any(p["number"] == pr["number"] for p in r["request"]["prs"])
    ]
    if not rows:
        raise ValueError(
            "legacy result requires a saved review; use --replan for an open issue"
        )
    request = rows[-1]["request"]
    saved_review(session, child, request)
    reviewed = next(p for p in request["prs"] if p["number"] == pr["number"])
    effective = reviewed
    if pr.get("merged_at"):
        effective = merged_evidence(session, child, pr, request)
    elif reviewed["head_sha"] != pr["head"]["sha"]:
        effective, _ = snapshot(pr)
    if reviewed["head_sha"] != effective["head_sha"]:
        resolved_snapshot(session, request, effective)
    validate_result(task, value["context"], record, pr, effective)
    if not pr.get("merged_at"):
        validate_worktree(Path(task["worktree"]), task["branch"])
        if (
            git("rev-parse", "HEAD", cwd=Path(task["worktree"])).decode().strip()
            != pr["head"]["sha"]
        ):
            raise ValueError("legacy result worktree HEAD differs from the reviewed PR")
    raw = session.gh.issue(child)
    if raw["state"] != "open" and (
        raw["state"] != "closed"
        or raw.get("state_reason") != "completed"
        or not pr.get("merged_at")
    ):
        raise ValueError("legacy closed issue requires a completed merge")
    value.update(phase="implementing", refresh_required=False)
    value["legacy_result"] = {
        "lineage_id": value["lineage_id"],
        "delegation_id": task["delegation_id"],
        "context_sha256": digest(value["context"]),
        "runtime": copy.deepcopy(value["runtime"]),
        "source": str(directory),
        "result_sha256": digest(record),
        "final_sha256": hashlib.sha256(
            (directory / "final.md").read_bytes()
        ).hexdigest(),
        "review_run_id": request["review_run_id"],
        "request_sha256": digest(request),
        "pr": pr["number"],
        "merged": bool(pr.get("merged_at")),
        "reviewed": copy.deepcopy(effective),
        "verification": copy.deepcopy(task["verification"]),
        "evidence": evidence,
        "invalidated": False,
    }
    task["review_preflight"] = {
        "review_run_id": request["review_run_id"],
        "context_sha256": digest(value["context"]),
        "invalidations": 0,
        "invalidated": False,
    }


def require_result(session: ParentSession, child: int, value: dict[str, Any]) -> None:
    """A legacy receipt only accepts its original result, never later commits."""
    from .worker_preflight import bind_runtime, latest_evidence

    receipt = value["legacy_result"]
    directory, record = latest_evidence(session, child)
    bind_runtime(value, record)
    if (
        str(directory) != receipt["source"]
        or digest(record) != receipt["result_sha256"]
        or hashlib.sha256((directory / "final.md").read_bytes()).hexdigest()
        != receipt["final_sha256"]
    ):
        raise ValueError("legacy worker result changed after migration")
    task = session.row["tasks"][str(child)]
    request = next(
        (
            r["request"]
            for r in session.row["reviews"]
            if r["request"]["review_run_id"] == receipt["review_run_id"]
        ),
        None,
    )
    if request is None or digest(request) != receipt["request_sha256"]:
        raise ValueError("legacy review receipt changed after migration")
    saved_review(session, child, request)
    pr = session.gh.get(f"repos/{session.gh.repo}/pulls/{receipt['pr']}")
    related = {
        item["number"]
        for item in session.gh.pulls({child})
        if child in closing_issues(item.get("body") or "", session.gh.repo)
    }
    if related != {receipt["pr"]}:
        raise ValueError("legacy result PR set changed; complete a new worker turn")
    validate_result(task, value["context"], record, pr, receipt["reviewed"])
    if receipt["merged"] and not pr.get("merged_at"):
        raise ValueError("legacy merge proof changed after migration")
    from .snapshot import git

    if not pr.get("merged_at") and (
        git("rev-parse", "HEAD", cwd=Path(task["worktree"])).decode().strip()
        != pr["head"]["sha"]
    ):
        raise ValueError("legacy result worktree HEAD changed after migration")
