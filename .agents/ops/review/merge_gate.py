#!/usr/bin/env python3
"""Explicit parent merge gate. Never launches a reviewer or polls for changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from issues import Issue  # noqa: E402
from merge_gate_policy import (  # noqa: E402
    MergeInput,
    MergeResult,
    authorization_revoked,
    auto_merge,
    confirmed_state,
    dependency_order,
    evaluate_parent_gate,
    publication_errors,
    shared_permission,
)
from parent.adopted_tasks import check_task_adoptions  # noqa: E402
from parent.execution import (  # noqa: E402
    authorization,
    require_run_authorization,
    requirement_body,
)
from parent.github import GitHub  # noqa: E402
from parent.policy import (  # noqa: E402
    closing_issues,
    digest,
    issue_number,
    nonempty,
    object_value,
)
from parent.session import ParentSession  # noqa: E402
from parent.snapshot import git, snapshot  # noqa: E402
from parent.state import Store, session_identity, state_directory  # noqa: E402
from parent_review_policy import (  # noqa: E402
    BaseAdvance,
    load_json,
    records,
    request_manifest,
    validate_result,
)
from parent_review_policy import snapshots as validate_snapshots  # noqa: E402
from review_policy import GateDecision, load_list_file, neutralize_markers  # noqa: E402
from review_workspace import (  # noqa: E402
    changed_paths,
    full_diff,
    verify_base_advances,
)


def read_json(path: Path) -> dict[str, Any]:
    return object_value(load_json(path.read_text(encoding="utf-8")))


def attempt_path(session: ParentSession, n: int, head: str) -> Path:
    return (
        session.store.directory
        / "merges"
        / digest(session.gh.repo.casefold())
        / f"pr-{n}-{head}.json"
    )


@dataclass
class Collection:
    parent: Issue
    children: dict[int, Issue]
    default_branch: str
    pulls: list[dict[str, Any]]
    candidates: list[int]
    details: dict[int, dict[str, Any]]
    snapshots: list[dict[str, Any]]
    paths: dict[int, tuple[str, ...]]
    fix_paths: dict[int, list[str]]
    fix_hashes: dict[int, str]
    errors: dict[int, list[str]]


def repository_default_branch(session: ParentSession) -> str:
    repo = object_value(session.gh.get(f"repos/{session.gh.repo}"))
    return nonempty(repo.get("default_branch"), "repository default_branch")


def check_adoptions(
    session: ParentSession,
    pr: dict[str, Any],
    family: tuple[Issue, dict[int, Issue]] | None = None,
) -> None:
    parent, children = family if family is not None else session.family()
    check_task_adoptions(session.gh, parent.body)
    for n in closing_issues(pr.get("body") or "", session.gh.repo) & set(children):
        check_task_adoptions(session.gh, children[n].body)


def collect(session: ParentSession, request: dict[str, Any] | None) -> Collection:
    parent, children = session.family()
    default_branch = repository_default_branch(session)
    pulls = session.gh.pulls()
    if len({p["number"] for p in pulls}) != len(pulls):
        raise ValueError("duplicate PR records from GitHub")
    targets = {p["number"]: p for p in request["prs"]} if request else {}
    issue_targets = set(children) or {parent.number}
    candidates = sorted(
        set(targets)
        | {
            p["number"]
            for p in pulls
            if p["state"] == "open"
            and closing_issues(p.get("body") or "", session.gh.repo) & issue_targets
        }
    )
    details, snapshots, paths, fix_paths, fix_hashes = {}, [], {}, {}, {}
    errors: dict[int, list[str]] = {}
    for n in candidates:
        raw = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
        if raw.get("number") != n:
            raise ValueError("GitHub PR identity mismatch")
        errors[n] = []
        try:
            check_adoptions(session, raw, (parent, children))
        except ValueError:
            errors[n].append("task_adoption_changed")
        if request is not None and n in targets and raw.get("merged_at"):
            # Recomputing after merge would change merge-base/diff. Preserve the
            # reviewed snapshot, or the one checked immediately before our merge.
            current = targets[n]
            journal = attempt_path(session, n, raw["head"]["sha"])
            if journal.exists():
                attempted = read_json(journal)
                if attempted.get("review_run_id") == request["review_run_id"]:
                    current = validate_snapshots([attempted["snapshot"]])[0]
                    if (
                        current["number"] != n
                        or current["head_sha"] != attempted["head_sha"]
                    ):
                        raise ValueError("merge journal snapshot identity mismatch")
            if raw["head"]["sha"] != current["head_sha"]:
                errors[n].append("merged_head_mismatch")
            if (
                raw["head"]["ref"] != current["branch"]
                or raw["base"]["ref"] != current["base_ref"]
                or raw["base"]["repo"]["full_name"].casefold()
                != session.gh.repo.casefold()
                or raw["head"]["repo"]["full_name"].casefold()
                != session.gh.repo.casefold()
            ):
                errors[n].append("merged_snapshot_identity_mismatch")
        else:
            current, _ = snapshot(raw)
        snapshots.append(current)
        paths[n] = tuple(changed_paths(current["merge_base_sha"], current["head_sha"]))
        if n in targets:
            before = targets[n]["head_sha"]
            try:
                git("merge-base", "--is-ancestor", before, current["head_sha"])
            except ValueError:
                errors[n].append("reviewed_head_ancestry_invalid")
            fix_paths[n] = changed_paths(before, current["head_sha"])
            fix_hashes[n] = hashlib.sha256(
                full_diff(before, current["head_sha"])
            ).hexdigest()
        raw["statusCheckRollup"] = []
        if raw["state"] == "open":
            ci = subprocess.run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(n),
                    "--repo",
                    session.gh.repo,
                    "--json",
                    "headRefOid,statusCheckRollup",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            data = object_value(load_json(ci.stdout))
            if set(data) != {"headRefOid", "statusCheckRollup"} or not isinstance(
                data["statusCheckRollup"], list
            ):
                raise ValueError("incomplete CI response")
            if data["headRefOid"] != current["head_sha"]:
                errors[n].append("head_changed_during_ci")
            raw["statusCheckRollup"] = data["statusCheckRollup"]
        details[n] = raw
    # Use the fresh individual PRs for dependency/duplicate checks, too.
    pulls = [details.get(p["number"], p) for p in pulls]
    return Collection(
        parent,
        children,
        default_branch,
        pulls,
        candidates,
        details,
        snapshots,
        paths,
        fix_paths,
        fix_hashes,
        errors,
    )


def load_evidence(
    session: ParentSession, request: dict[str, Any] | None, view: Collection
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    str | None,
    tuple[str, ...],
    dict[str, Any] | None,
]:
    if request is None:
        return None, [], None, ("parent_review_missing",), None
    run_dir = session.store.directory / "reviews" / request["review_run_id"]
    if (
        not (run_dir / "result.json").exists()
        or not (run_dir / "execution.json").exists()
    ):
        return None, [], None, ("parent_review_missing_or_interrupted",), None
    result = validate_result(read_json(run_dir / "result.json"))
    execution = read_json(run_dir / "execution.json")
    if result["request"] != request or read_json(run_dir / "request.json") != request:
        raise ValueError("local review result/request mismatch")
    auth = authorization(session, request)
    parent_comments = session.gh.comments(session.parent)
    pr_comments = {n: session.gh.comments(n) for n in view.candidates}
    errors = list(
        publication_errors(
            result, execution, parent_comments, pr_comments, session.login, auth
        )
    )
    if request.get("authorization", auth) != auth:
        errors.append("review_execution_basis_changed")
    if request["schema_version"] == 2:
        for n, expected in request["requirements"].items():
            raw = session.gh.issue(int(n))
            if digest(requirement_body(raw.get("body") or "")) != expected:
                reason = f"issue_{n}_reviewed_requirements_changed"
                affected = [
                    p["number"] for p in result["prs"] if int(n) in p["child_issues"]
                ]
                if int(n) == session.parent or not affected:
                    errors.append(reason)
                else:
                    for pr in affected:
                        view.errors.setdefault(pr, []).append(reason)
    if execution.get("session") != session.session:
        errors.append("review_owner_changed")
    state = confirmed_state(session.history(), request, auth)
    permission = shared_permission(parent_comments, session.login, request, auth)
    resolutions = []
    for record in records(parent_comments, session.login):
        if (
            record["kind"] != "resolution"
            or record["review_run_id"] != request["review_run_id"]
        ):
            continue
        key = digest(record)
        saved_path = run_dir / f"resolution-{key}.json"
        diff_path = run_dir / f"resolution-{key}.diff"
        # A half-published resolution is never sufficient to release a finding.
        if not saved_path.exists() or not diff_path.exists():
            continue
        saved = read_json(saved_path)
        n = record["pr"]
        if (
            saved.get("published") is not True
            or saved.get("record") != record
            or record not in records(pr_comments.get(n, []), session.login)
            or hashlib.sha256(diff_path.read_bytes()).hexdigest()
            != record["fix_diff_sha256"]
        ):
            continue
        current = next((p for p in view.snapshots if p["number"] == n), None)
        if (
            current
            and record["fixed_snapshot"]["head_sha"] == current["head_sha"]
            and record["fix_diff_sha256"] != view.fix_hashes.get(n)
        ):
            view.errors[n].append("resolution_fix_diff_mismatch")
            continue
        resolutions.append(record)
    return result, resolutions, state, tuple(errors), permission


def evaluate(
    session: ParentSession, request: dict[str, Any] | None, view: Collection
) -> dict[int, MergeResult]:
    result, resolutions, state, errors, permission = load_evidence(
        session, request, view
    )
    advances: set[BaseAdvance] = set()
    # A merged peer may explain base history only after its confirmed head and
    # identity have been checked. A mismatch invalidates that evidence for all.
    errors = (
        *errors,
        *(
            f"pr_{n}_{reason}"
            for n in view.candidates
            if view.details[n].get("merged_at")
            for reason in view.errors[n]
        ),
    )
    if result is not None and not errors:
        targets = {p["number"]: p for p in result["request"]["prs"]}
        peers = [
            view.details[n]
            for n in view.candidates
            if n in targets and view.details[n].get("merged_at")
        ]
        for current in view.snapshots:
            n = current["number"]
            if n not in targets or current["base_sha"] == targets[n]["base_sha"]:
                continue
            try:
                advances.update(verify_base_advances(result, [current], peers))
            except ValueError:
                # Keep this PR's failure explicit without blocking an unrelated
                # eligible PR. Coverage also reports the changed snapshot fields.
                view.errors[n].append("base_advance_unverified")
    required = load_list_file(HERE / "required_checks.txt")
    patterns = load_list_file(HERE / "shared_paths.txt")
    for n in view.candidates:
        pr = view.details[n]
        if (
            pr["state"] == "open"
            and attempt_path(session, n, pr["head"]["sha"]).exists()
        ):
            view.errors[n].append("merge_already_attempted")
    return {
        n: evaluate_parent_gate(
            MergeInput(
                repo=session.gh.repo,
                default_branch=view.default_branch,
                parent_issue=session.parent,
                pr=view.details[n],
                children=view.children,
                tasks=session.row["tasks"],
                pulls=view.pulls,
                result=result,
                current=view.snapshots,
                resolutions=resolutions,
                fix_paths=view.fix_paths,
                fix_diff_digests=view.fix_hashes,
                changed_paths=view.paths[n],
                review_state=state,
                evidence_errors=(*errors, *view.errors[n]),
                required_checks=required,
                shared_patterns=patterns,
                base_advances=tuple(advances),
                permission=permission,
                parent_needs_human="needs-human" in view.parent.labels,
            )
        )
        for n in view.candidates
    }


def completion(
    session: ParentSession, pr: dict[str, Any], *, write: bool
) -> tuple[str, ...]:
    if not pr.get("merged_at"):
        return ("merge_not_confirmed",)
    if pr["base"]["ref"] != repository_default_branch(session):
        return ("base_not_default_branch",)
    parent, children = session.family()
    issue_targets = set(children) or {parent.number}
    linked = closing_issues(pr.get("body") or "", session.gh.repo) & issue_targets
    if not linked:
        return ("merged_pr_child_missing",)
    for child in sorted(linked):
        raw = session.gh.issue(child)
        if raw["state"] != "closed" or raw.get("state_reason") != "completed":
            return (f"child_{child}_close_not_confirmed",)
    if write and children:
        # Parent progress is derived from GitHub, never from the merge command's exit.
        session.reconcile()
        session.sync_progress()
    return ()


def merge(session: ParentSession, enabled: bool) -> dict[str, Any]:
    session.agreed()
    request = (
        request_manifest(session.row["reviews"][-1]["request"], instructed=False)
        if session.row["reviews"]
        else None
    )
    results: dict[int, dict[str, Any]] = {}
    evaluations = []
    handled: set[int] = set()
    while True:
        session.agreed()
        view = collect(session, request)
        decisions = evaluate(session, request, view)
        order = dependency_order(
            [view.details[n] for n in view.candidates], view.children, session.gh.repo
        )
        made_progress = False
        for n in order:
            if n in handled:
                continue
            pr = view.details[n]
            entry = {
                "pr": n,
                **asdict(decisions[n]),
                "snapshot": next(s for s in view.snapshots if s["number"] == n),
                "action": "dry_run"
                if not enabled
                else "eligible"
                if decisions[n].decision == GateDecision.MERGE
                else "blocked",
            }
            if pr.get("merged_at"):
                errors = (*view.errors[n], *completion(session, pr, write=enabled))
                entry.update(
                    decision="blocked" if errors else "merge",
                    reasons=errors,
                    action="merge_unconfirmed" if errors else "already_merged",
                )
                results[n] = entry
                evaluations.append(entry)
                handled.add(n)
                continue
            results[n] = entry
            evaluations.append(entry)
            if decisions[n].decision != GateDecision.MERGE or not enabled:
                continue
            # Re-read all coverage, base/head/diff, CI, parent state and dependencies
            # immediately before the write. --match-head-commit closes the last race.
            session.agreed()
            fresh = collect(session, request)
            decision = evaluate(session, request, fresh)[n]
            current = next(s for s in fresh.snapshots if s["number"] == n)
            entry = {
                "pr": n,
                **asdict(decision),
                "snapshot": current,
                "action": "blocked",
            }
            if current["head_sha"] != pr["head"]["sha"]:
                entry.update(decision="blocked", reasons=("head_changed_before_merge",))
            results[n] = entry
            evaluations.append(entry)
            if entry["decision"] != GateDecision.MERGE:
                continue
            check_adoptions(session, fresh.details[n])
            journal_path = attempt_path(session, n, current["head_sha"])
            if journal_path.exists():
                entry.update(decision="blocked", reasons=("merge_already_attempted",))
                continue
            attempt = {
                "pr": n,
                "head_sha": current["head_sha"],
                "review_run_id": request["review_run_id"] if request else None,
                "snapshot": current,
                "status": "attempting",
            }
            Store.write_json(journal_path, attempt)
            command = subprocess.run(
                [
                    "gh",
                    "pr",
                    "merge",
                    str(n),
                    "--repo",
                    session.gh.repo,
                    "--merge",
                    "--match-head-commit",
                    current["head_sha"],
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            attempt["exit_code"] = command.returncode
            actual = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
            errors = completion(
                session, actual, write=actual.get("merged_at") is not None
            )
            if actual["head"]["sha"] != current["head_sha"]:
                errors = (*errors, "merge_head_mismatch")
            if command.returncode and not actual.get("merged_at"):
                errors = (*errors, "merge_command_failed")
            attempt["status"] = "unconfirmed" if errors else "merged"
            Store.write_json(journal_path, attempt)
            entry.update(
                action="merge_unconfirmed" if errors else "merged",
                decision="blocked" if errors else "merge",
                reasons=errors,
            )
            handled.add(n)
            made_progress = actual.get("merged_at") is not None
            # No reuse of the pre-merge collection for any remaining PR.
            if made_progress:
                break
        if not made_progress or not view.children:
            break
    return {
        "parent": session.parent,
        "review_run_id": request["review_run_id"] if request else None,
        "auto_merge": enabled,
        "agent_invocations": 0,
        "reasons": [] if results else ["no_target_prs"],
        "results": [results[n] for n in sorted(results)],
        "evaluations": evaluations,
    }


def record_permission(
    session: ParentSession, allowed: bool, evidence_file: Path
) -> dict[str, Any]:
    session.agreed()
    if not session.row["reviews"]:
        raise ValueError("a fixed parent review run is required")
    request = request_manifest(session.row["reviews"][-1]["request"], instructed=False)
    evidence = nonempty(
        evidence_file.read_text(encoding="utf-8"), "human agreement evidence"
    )
    history = session.history()
    prior = [e for e in history if e["event"] == "merge_permission"]
    recent: list[dict[str, Any]] = []
    auth = require_run_authorization(session, request, required_issues=set())
    for event in history:
        if authorization_revoked(event, auth):
            recent = []
        elif event["event"] == "merge_permission":
            recent.append(event)
    record = {
        "event": "merge_permission",
        "review_run_id": request["review_run_id"],
        "request_sha256": digest(request),
        **auth,
        "allow_shared_foundation": allowed,
        "evidence": evidence,
    }
    if recent and all(recent[-1].get(k) == v for k, v in record.items()):
        return {"parent": session.parent, **record, "revision": recent[-1]["revision"]}
    # A→deny→A must be a new event even when the human evidence text is reused.
    record["revision"] = max((e.get("revision", 0) for e in prior), default=0) + 1
    session.post(
        record,
        f"共有基盤変更を含む親mergeの明示許可: {allowed}\n\n人間との合意根拠:\n{neutralize_markers(evidence)}",
    )
    return {"parent": session.parent, **record}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repo")
    root.add_argument("--parent", required=True)
    root.add_argument("--state-dir", type=Path, default=state_directory())
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("merge")
    permission = commands.add_parser("shared-permission")
    choice = permission.add_mutually_exclusive_group(required=True)
    choice.add_argument("--allow-shared-foundation", action="store_true")
    choice.add_argument("--deny-shared-foundation", action="store_true")
    permission.add_argument("--evidence-file", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        enabled = auto_merge(os.environ.get("AGENT_AUTO_MERGE"))
        identity = session_identity(dict(os.environ))
        gh = GitHub(args.repo)
        with Store(args.state_dir).locked() as store:
            session = ParentSession(
                gh, store, issue_number(args.parent, gh.repo), identity
            )
            value = (
                merge(session, enabled)
                if args.command == "merge"
                else record_permission(
                    session, args.allow_shared_foundation, args.evidence_file
                )
            )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return (
            0
            if args.command != "merge"
            or (
                value["results"]
                and all(r["decision"] == "merge" for r in value["results"])
            )
            else 1
        )
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        StopIteration,
        subprocess.SubprocessError,
    ) as exc:
        # Do not emit arbitrary GitHub payloads or private registry content.
        print(
            json.dumps(
                {
                    "decision": "blocked",
                    "reasons": ["input_or_collection_error"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "agent_invocations": 0,
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
