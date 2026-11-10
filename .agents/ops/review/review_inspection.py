"""Observe review evidence and input differences. No request or permission is created."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from merge_gate_policy import publication_errors, python_ci_errors
from parent.github import GitHub
from parent.policy import (
    closing_issues,
    digest,
    events,
    number,
    object_value,
    review_state,
)
from parent.session import ParentSession
from parent.snapshot import snapshot
from parent.state import Store
from parent_review_policy import (
    evaluate_coverage,
    execution_stages,
    load_json,
    pr_findings,
    request_manifest,
    superseding_reviews,
    validate_result,
)
from review_evidence import followups, python_check_runs
from review_interruption import CLOSED, exit_confirmation_record, interruption_record
from review_observations import baseline, compare_observation, discussion
from review_workspace import changed_paths, full_diff, verify_rebased_snapshot

T = TypeVar("T")
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError)


def read_json(path: Path) -> dict[str, Any]:
    return dict(object_value(load_json(path.read_text(encoding="utf-8"))))


def attempt(
    errors: list[dict[str, str]],
    scope: str,
    read: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    try:
        return read(*args, **kwargs)
    except READ_ERRORS as exc:
        # Exception strings can contain private state paths and runtime identities.
        # A scoped failure is explicit; it must not become empty successful data.
        errors.append({"scope": scope, "error_type": type(exc).__name__})
        return None


def read_pr(gh: GitHub, n: int) -> dict[str, Any]:
    raw: dict[str, Any] = object_value(gh.get(f"repos/{gh.repo}/pulls/{n}"))
    if raw.get("number") != n:
        raise ValueError("PR identity mismatch")
    if (
        not isinstance(raw.get("title"), str)
        or (raw.get("body") is not None and not isinstance(raw["body"], str))
        or raw.get("state") not in {"open", "closed"}
        or type(raw.get("draft")) is not bool
    ):
        raise ValueError("incomplete PR fields")
    for side in ("head", "base"):
        if not isinstance(raw[side]["repo"]["full_name"], str):
            raise ValueError("PR repository identity missing")
        if not isinstance(raw[side]["ref"], str) or not isinstance(
            raw[side]["sha"], str
        ):
            raise ValueError("PR refs missing")
    return raw


def read_discussion(
    session: ParentSession, kind: str, n: int, raw: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return discussion(session, kind, raw if raw is not None else session.gh.issue(n))


def pr_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: raw.get(key)
        for key in ("number", "title", "body", "state", "draft", "merged_at")
    } | {
        side: {
            "sha": raw[side]["sha"],
            "ref": raw[side]["ref"],
            "repo": raw[side]["repo"]["full_name"],
        }
        for side in ("head", "base")
    }


def compare_snapshot(
    original: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    # snapshot performs the existing bounded fetch for missing immutable objects.
    observed, _ = snapshot(
        {
            "number": original["number"],
            "head": {"sha": original["head_sha"], "ref": original["branch"]},
            "base": {"sha": original["base_sha"], "ref": original["base_ref"]},
        }
    )
    if observed != original:
        raise ValueError("reviewed snapshot differs from actual git evidence")
    return {
        "fields": {
            key: {
                "before": original[key],
                "after": current[key],
                "equal": original[key] == current[key],
            }
            for key in sorted(original)
        },
        "changed_paths": changed_paths(original["head_sha"], current["head_sha"]),
        "reviewed_paths": changed_paths(
            original["merge_base_sha"], original["head_sha"]
        ),
        "fix_diff_sha256": hashlib.sha256(
            full_diff(original["head_sha"], current["head_sha"])
        ).hexdigest(),
        "rebase_inheritance_verified": verify_rebased_snapshot(original, current),
    }


def inspect_run(
    session: ParentSession,
    row: Mapping[str, Any],
    reviews: list[dict[str, Any]],
    n: int,
    current: Mapping[str, Any] | None,
    observations: Mapping[str, Any],
    comments: Callable[[int], list[dict[str, Any]]],
    parent_comments: list[dict[str, Any]],
) -> dict[str, Any]:
    request = request_manifest(row["request"], instructed=False)
    run = request["review_run_id"]
    original = next(p for p in request["prs"] if p["number"] == n)
    directory = session.store.directory / "reviews" / run
    position = next(
        i for i, r in enumerate(reviews) if r["request"]["review_run_id"] == run
    )
    errors: list[dict[str, str]] = []
    entry: dict[str, Any] = {
        "review_run_id": run,
        "request_sha256": digest(request),
        "review_state": attempt(
            errors,
            "review_state",
            lambda: review_state(
                events(parent_comments, session.login, session.parent), request
            ),
        ),
        "snapshot": original,
        "reason": request["reason"],
        "subsequent_requests": [
            r["request"]["review_run_id"]
            for r in reviews[position + 1 :]
            if any(p["number"] == n for p in r["request"]["prs"])
        ],
        "errors": errors,
        "result": None,
        "coverage_check": None,
        "saved_output": None,
    }
    saved_request = attempt(
        errors, "local_request", lambda: read_json(directory / "request.json")
    )
    if saved_request is not None and saved_request != request:
        errors.append({"scope": "local_request", "error_type": "RequestMismatch"})
    execution = (
        attempt(errors, "execution", lambda: read_json(directory / "execution.json"))
        if (directory / "execution.json").exists()
        else None
    )
    entry["execution"] = None
    runtime_path = directory / "agent-runtime/result.json"
    runtime = (
        attempt(errors, "runtime", read_json, runtime_path)
        if runtime_path.exists()
        else None
    )
    entry["runtime"] = (
        {
            key: runtime.get(key)
            for key in (
                "result",
                "exit_code",
                "cli_exit_code",
                "turn_end_reason",
                "cleanup_complete",
                "agent_spawned",
            )
        }
        if runtime is not None
        else None
    )
    if execution is not None:
        stages = attempt(errors, "stages", lambda: execution_stages(execution))
        closure = attempt(
            errors, "interruption", interruption_record, execution, request
        )
        entry["execution"] = {
            "status": execution.get("status"),
            "agent_invocations": execution.get("agent_invocations"),
            "interruption": closure,
            "exit_confirmation": attempt(
                errors,
                "exit_confirmation",
                exit_confirmation_record,
                execution,
                request,
            ),
            "stages": {
                name: {
                    k: value
                    for k, value in stage.items()
                    if k
                    in {
                        "status",
                        "cleanup_complete",
                        "workspace_cleanup_complete",
                        "exit_code",
                    }
                }
                for name, stage in (stages or {}).items()
            },
        }
    comparison = (
        attempt(
            errors, "snapshot_comparison", lambda: compare_snapshot(original, current)
        )
        if current
        else None
    )
    entry["comparison"] = comparison
    context = (
        attempt(errors, "context", lambda: read_json(directory / "context.json"))
        if (directory / "context.json").exists()
        else None
    )
    before = (
        attempt(errors, "observation_baseline", lambda: baseline(context, session))
        if context
        else None
    )
    entry["input_changes"] = {
        key: attempt(
            errors,
            f"input_comparison:{key}",
            compare_observation,
            (before or {}).get(key),
            value,
        )
        for key, value in observations.items()
    }
    entry["recorded_requirements"] = request.get("requirements")
    if not (directory / "result.json").exists():
        if execution is not None and execution.get("status") in {
            "published",
            "publication_failed",
        }:
            errors.append({"scope": "result", "error_type": "PublishedResultMissing"})
        return entry
    result = attempt(
        errors, "result", lambda: validate_result(read_json(directory / "result.json"))
    )
    if result is None:
        return entry
    if result["request"] != request:
        errors.append({"scope": "result", "error_type": "RequestMismatch"})
        return entry
    reviewed = next(p for p in result["prs"] if p["number"] == n)
    entry["result"] = {
        "retained_only": (execution or {}).get("status") == CLOSED,
        "kind": result["kind"],
        "mode": result.get("mode"),
        "source_run": result.get("source_run"),
        "status": result["status"],
        "verdict": result["verdict"],
        "result_sha256": digest(result),
        "coverage": reviewed["coverage"],
        "child_issues": reviewed["child_issues"],
        "summary": reviewed["summary"],
        "findings": pr_findings(result, n),
        "evidence_waits": reviewed.get("evidence_waits", []),
        "error": result["error"],
    }
    if result["kind"] == "adoption":
        from review_recovery import verify_adoption

        attempt(
            errors,
            "adoption_source",
            lambda: verify_adoption(session.store.directory, result),
        )
        superseded = attempt(
            errors,
            "adoption_registry",
            superseding_reviews,
            reviews,
            result["source_run"],
            request,
        )
        entry["superseding_source_requests"] = (
            superseded.get(n, []) if superseded is not None else None
        )
    # The publication checker is shared with gate. Authority here is historical,
    # never a grant to the viewer. Per-PR failures remain explicitly scoped.
    public_comments = {}
    unknown_prs = []
    for target in result["prs"]:
        m = target["number"]
        obtained = attempt(errors, f"publication:pr:{m}", comments, m)
        if obtained is None:
            unknown_prs.append(m)
        else:
            public_comments[m] = obtained
    entry["publication_unknown_prs"] = unknown_prs
    public = attempt(
        errors,
        "publication",
        lambda: publication_errors(
            result,
            execution or {},
            parent_comments,
            public_comments,
            session.login,
            request.get("authorization", (execution or {}).get("authorization", {})),
        ),
    )
    entry["publication_errors"] = public
    entry["projection_errors"] = (
        [e for e in public if e.startswith(f"pr_{n}_")] if public is not None else None
    )
    entry["parent_run_url"] = (execution or {}).get("parent_run_url")
    proofs = attempt(
        errors,
        "followups",
        lambda: followups(
            directory,
            result,
            parent_comments,
            public_comments,
            session.login,
            {n: current} if current else {},
            {n: comparison["fix_diff_sha256"]} if comparison else {},
            selected={n},
        ),
    )
    own_proofs = [p for p in proofs or [] if p["record"]["pr"] == n]
    entry["followups"] = [
        {
            **p,
            "result_matches": p["record"]["result_sha256"] == digest(result),
            "snapshot_matches": (
                p["record"].get("fixed_snapshot") or p["record"].get("snapshot")
            )
            == current,
            "rebase_inheritance_verified": attempt(
                errors,
                "followup_inheritance",
                verify_rebased_snapshot,
                p["record"].get("fixed_snapshot") or p["record"].get("snapshot"),
                current,
            )
            if current
            else None,
        }
        for p in own_proofs
    ]
    if (
        current
        and comparison
        and proofs is not None
        and (execution or {}).get("status") != CLOSED
    ):
        inherited = set()
        for old in [
            original,
            *[
                p["record"].get("fixed_snapshot", p["record"].get("snapshot"))
                for p in own_proofs
                if not p["errors"]
            ],
        ]:
            if verify_rebased_snapshot(old, current):
                inherited.add((digest(old), digest(current)))
        coverage = evaluate_coverage(
            result,
            [current],
            [p["record"] for p in own_proofs if not p["errors"]],
            {n: comparison["changed_paths"]},
            fix_diff_digests={n: comparison["fix_diff_sha256"]},
            inherited_snapshots=frozenset(inherited),
        )
        entry["coverage_check"] = asdict(coverage)
    else:
        entry["coverage_check"] = None
    if (directory / "output.json").exists():
        candidate = attempt(
            errors,
            "saved_output",
            lambda: validate_result(read_json(directory / "output.json")),
        )
        if candidate is not None:
            entry["saved_output"] = {
                "status": "candidate_only",
                "verdict": candidate["verdict"],
                "prs": candidate["prs"],
                "cross_findings": candidate["cross_findings"],
            }
    return entry


def inspect_reviews(
    gh: GitHub, store: Store, parent: int, selected: list[int]
) -> dict[str, Any]:
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("review-inspect requires unique PR numbers")
    selected = sorted(number(n) for n in selected)
    initial = store.read_data()
    if (
        set(initial) != {"version", "parents"}
        or type(initial["version"]) is not int
        or initial["version"] not in {1, 2}
    ):
        raise ValueError("unknown parent registry schema")
    store.data = initial  # Private in-memory read only; never save or migrate.
    keys = [
        k
        for k in object_value(initial["parents"])
        if k.casefold() == f"{gh.repo.casefold()}#{parent}"
    ]
    if len(keys) > 1:
        raise ValueError("ambiguous parent registry aliases")
    saved = initial["parents"].get(keys[0], {}) if keys else {}
    reviews = saved.get("reviews", [])
    ids = []
    for row in reviews:
        request = request_manifest(row["request"], instructed=False)
        if (
            request["repo"].casefold() != gh.repo.casefold()
            or request["parent_issue"] != parent
        ):
            raise ValueError("review registry target identity mismatch")
        ids.append(request["review_run_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate review run IDs")
    session = ParentSession(gh, store, parent, None)
    root_errors: list[dict[str, str]] = []
    gh.issue(parent)
    children = {
        r["number"] for r in gh.rows(f"repos/{gh.repo}/issues/{parent}/sub_issues")
    }
    parent_comments = attempt(
        root_errors, "parent_comments", lambda: gh.comments(parent)
    )
    comment_cache: dict[int, list[dict[str, Any]]] = {}
    if parent_comments is not None:
        comment_cache[parent] = parent_comments

    def comments(n: int) -> list[dict[str, Any]]:
        if n not in comment_cache:
            comment_cache[n] = gh.comments(n)
        return comment_cache[n]

    output: dict[str, Any] = {
        "schema_version": 1,
        "repo": gh.repo,
        "parent_issue": parent,
        "observed_at": datetime.now(UTC).isoformat(),
        "agent_invocations": 0,
        "errors": root_errors,
        "prs": [],
    }
    for n in selected:
        errors: list[dict[str, str]] = []
        pr: dict[str, Any] = {
            "number": n,
            "errors": errors,
            "history": [],
            "identity": None,
            "current_snapshot": None,
            "python_ci": None,
            "changed_paths": None,
        }
        output["prs"].append(pr)
        raw = attempt(errors, "pr", read_pr, gh, n)
        current = None
        observations = {}
        if raw is not None and raw.get("number") != n:
            errors.append({"scope": "pr", "error_type": "IdentityMismatch"})
            raw = None
        if raw is not None:
            linked = closing_issues(raw.get("body") or "", gh.repo) & (
                children or {parent}
            )
            pr["identity"] = {
                "head_repo": raw["head"]["repo"]["full_name"],
                "base_repo": raw["base"]["repo"]["full_name"],
                "child_issues": sorted(linked),
                "state": raw["state"],
                "draft": raw["draft"],
            }
            if not linked or any(
                raw[side]["repo"]["full_name"].casefold() != gh.repo.casefold()
                for side in ("head", "base")
            ):
                errors.append({"scope": "pr", "error_type": "TargetIdentityMismatch"})
            frozen = attempt(errors, "snapshot", snapshot, raw)
            current = frozen[0] if frozen else None
            for kind, target in [
                ("issue", parent),
                *[("issue", c) for c in sorted(linked - {parent})],
                ("pr", n),
            ]:
                item = attempt(
                    errors,
                    f"discussion:{kind}:{target}",
                    read_discussion,
                    session,
                    kind,
                    target,
                    raw if kind == "pr" else None,
                )
                if item is not None:
                    observations[f"{kind}:{target}"] = item
            pr["current_snapshot"] = current
            if current is not None:
                ci = attempt(
                    errors, "python_ci", python_check_runs, session, current["head_sha"]
                )
                pr["python_ci"] = (
                    {
                        "head_sha": current["head_sha"],
                        "check_runs": ci,
                        "errors": python_ci_errors(ci, current["head_sha"]),
                    }
                    if ci is not None
                    else None
                )
                pr["changed_paths"] = attempt(
                    errors,
                    "changed_paths",
                    changed_paths,
                    current["merge_base_sha"],
                    current["head_sha"],
                )
        relevant = [
            r for r in reviews if any(p["number"] == n for p in r["request"]["prs"])
        ]
        pr["latest_request"] = (
            relevant[-1]["request"]["review_run_id"] if relevant else None
        )
        for row in relevant:
            item = attempt(
                errors,
                f"run:{row['request']['review_run_id']}",
                inspect_run,
                session,
                row,
                reviews,
                n,
                current,
                observations,
                comments,
                parent_comments or [],
            )
            if item is not None:
                pr["history"].append(item)
        # Re-read the selected PR and discussions, not unrelated family PRs.
        # This is an observation window, never a reservation for a later action.
        if raw is not None:
            fresh = attempt(errors, "postcheck:pr", read_pr, gh, n)
            if fresh is not None and pr_observation(fresh) != pr_observation(raw):
                errors.append(
                    {"scope": "postcheck:pr", "error_type": "ChangedDuringInspection"}
                )
            for key, before in observations.items():
                kind, target_text = key.split(":")
                target = int(target_text)
                after = (
                    attempt(
                        errors,
                        f"postcheck:{key}",
                        read_discussion,
                        session,
                        kind,
                        target,
                        fresh if kind == "pr" else None,
                    )
                    if kind != "pr" or fresh is not None
                    else None
                )
                if after is not None and after != before:
                    errors.append(
                        {
                            "scope": f"postcheck:{key}",
                            "error_type": "ChangedDuringInspection",
                        }
                    )
    final = store.read_data()
    if {k: v for k, v in final["parents"].items() if k.casefold() == session.key} != {
        k: v for k, v in initial["parents"].items() if k.casefold() == session.key
    }:
        root_errors.append(
            {"scope": "registry", "error_type": "ChangedDuringInspection"}
        )
    return output
