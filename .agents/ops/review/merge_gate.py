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
sys.path.insert(0, str(HERE.parent / "lib"))

from issues import Issue  # noqa: E402
from merge_gate_policy import (  # noqa: E402
    MergeInput,
    MergeResult,
    auto_merge,
    confirmed_state,
    dependency_order,
    evaluate_parent_gate,
    publication_errors,
)
from parent.adopted_tasks import check_task_adoptions  # noqa: E402
from parent.execution import (  # noqa: E402
    authorization,
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
from parent.snapshot import snapshot  # noqa: E402
from parent.state import Store, session_identity, state_directory  # noqa: E402
from parent.work_targets import dependency_graph, require_separate_self_prs  # noqa: E402
from parent_review_policy import (  # noqa: E402
    BaseAdvance,
    load_json,
    request_manifest,
    superseding_reviews,
    validate_result,
)
from parent_review_policy import snapshots as validate_snapshots  # noqa: E402
from review_evidence import followups, python_check_runs  # noqa: E402
from review_interruption import interruption_record  # noqa: E402
from review_policy import GateDecision  # noqa: E402
from review_recovery import verify_adoption  # noqa: E402
from review_workspace import (  # noqa: E402
    changed_paths,
    full_diff,
    verify_rebased_snapshot,
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
    parent, children = family if family is not None else session.work_items()
    check_task_adoptions(session.gh, parent.body)
    for n in closing_issues(pr.get("body") or "", session.gh.repo) & set(children):
        check_task_adoptions(session.gh, children[n].body)


def collect(session: ParentSession, request: dict[str, Any] | None) -> Collection:
    with session.gh.fresh_reads():
        return _collect(session, request)


def _collect(session: ParentSession, request: dict[str, Any] | None) -> Collection:
    parent, children = session.work_items()
    policy_children, external_deps = dependency_graph(parent, children)
    dependency_errors = []
    for dep in external_deps:
        raw = session.gh.issue(dep)
        if raw["state"] != "closed" or raw.get("state_reason") != "completed":
            dependency_errors.append(f"dependency_{dep}_not_completed")
    default_branch = repository_default_branch(session)
    pulls = session.gh.pulls()
    require_separate_self_prs(parent, pulls, session.gh.repo)
    if len({p["number"] for p in pulls}) != len(pulls):
        raise ValueError("duplicate PR records from GitHub")
    targets = {p["number"]: p for p in request["prs"]} if request else {}
    issue_targets = set(children)
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
        errors[n] = list(dependency_errors)
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
            fix_paths[n] = changed_paths(before, current["head_sha"])
            fix_hashes[n] = hashlib.sha256(
                full_diff(before, current["head_sha"])
            ).hexdigest()
        raw["statusCheckRollup"] = []
        raw["ci_head_sha"] = None
        raw["ci_check_runs"] = []
        if raw["state"] == "open":
            with session.store.unlocked_read():
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
            raw["ci_head_sha"] = data["headRefOid"]
            raw["ci_check_runs"] = python_check_runs(session, current["head_sha"])
        details[n] = raw
    # Use the fresh individual PRs for dependency/duplicate checks, too.
    pulls = [details.get(p["number"], p) for p in pulls]
    return Collection(
        parent,
        policy_children,
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
]:
    if request is None:
        return None, [], None, ("parent_review_missing",)
    run_dir = session.store.directory / "reviews" / request["review_run_id"]
    if (run_dir / "execution.json").exists():
        execution = read_json(run_dir / "execution.json")
        if interruption_record(execution, request) is not None:
            if read_json(run_dir / "request.json") != request:
                raise ValueError("local closed review/request mismatch")
            return None, [], None, ("parent_review_interrupted_closed",)
    if (
        not (run_dir / "result.json").exists()
        or not (run_dir / "execution.json").exists()
    ):
        return None, [], None, ("parent_review_missing_or_interrupted",)
    result = validate_result(read_json(run_dir / "result.json"))
    execution = read_json(run_dir / "execution.json")
    if result["request"] != request or read_json(run_dir / "request.json") != request:
        raise ValueError("local review result/request mismatch")
    from review_recovery import verify_result_runtime

    verify_result_runtime(run_dir, result, execution)
    auth = authorization(session, request)
    if result["kind"] == "adoption":
        source = next(
            (
                r["request"]
                for r in session.row["reviews"]
                if r["request"]["review_run_id"] == result["source_run"]
            ),
            None,
        )
        if source is None or source["prs"] != request["prs"]:
            raise ValueError("adoption has no registered source review")
        if (
            read_json(
                session.store.directory
                / "reviews"
                / result["source_run"]
                / "request.json"
            )
            != source
        ):
            raise ValueError("adoption source request differs from registry")
        verify_adoption(session.store.directory, result)
        for n in superseding_reviews(
            session.row["reviews"], result["source_run"], request
        ):
            if n in view.errors:
                view.errors[n].append("adoption_source_superseded")
    parent_comments = session.gh.comments(session.parent)
    pr_comments = {n: session.gh.comments(n) for n in view.candidates}
    errors = list(
        publication_errors(
            result,
            execution,
            parent_comments,
            pr_comments,
            session.login,
            request.get("authorization", auth),
        )
    )
    state = confirmed_state(session.history(), request, auth)
    resolutions = []
    for proof in followups(
        run_dir,
        result,
        parent_comments,
        pr_comments,
        session.login,
        {p["number"]: p for p in view.snapshots},
        view.fix_hashes,
    ):
        if proof["errors"] == ["resolution_fix_diff_mismatch"]:
            view.errors[proof["record"]["pr"]].append("resolution_fix_diff_mismatch")
        if not proof["errors"]:
            resolutions.append(proof["record"])
    return result, resolutions, state, tuple(errors)


def evaluate(
    session: ParentSession, request: dict[str, Any] | None, view: Collection
) -> dict[int, MergeResult]:
    result, resolutions, state, errors = load_evidence(session, request, view)
    advances: set[BaseAdvance] = set()
    inherited: set[tuple[str, str]] = set()
    if result is not None:
        originals = {p["number"]: p for p in result["request"]["prs"]}
        for current in view.snapshots:
            n = current["number"]
            candidates = [originals[n]] if n in originals else []
            candidates += [
                r["fixed_snapshot"] if r["kind"] == "resolution" else r["snapshot"]
                for r in resolutions
                if r["pr"] == n
            ]
            for original in candidates:
                if verify_rebased_snapshot(original, current):
                    inherited.add((digest(original), digest(current)))
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
                base_advances=tuple(advances),
                inherited_snapshots=frozenset(inherited),
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
    parent, children = session.work_items()
    issue_targets = set(children)
    linked = closing_issues(pr.get("body") or "", session.gh.repo) & issue_targets
    if not linked:
        return ("merged_pr_child_missing",)
    for child in sorted(linked):
        raw = session.gh.issue(child)
        if raw["state"] != "closed" or raw.get("state_reason") != "completed":
            return (f"child_{child}_close_not_confirmed",)
    if write and parent.children:
        # Parent progress is derived from GitHub, never from the merge command's exit.
        session.reconcile()
        session.sync_progress()
    return ()


def merge(session: ParentSession, enabled: bool) -> dict[str, Any]:
    session.agreed()
    runs = [
        request_manifest(row["request"], instructed=False)
        for row in session.row["reviews"]
    ]
    if not runs:
        return _merge_request(session, enabled, None, None)
    latest = {
        pr["number"]: request["review_run_id"]
        for request in runs
        for pr in request["prs"]
    }
    results: dict[int, dict[str, Any]] = {}
    evaluations = []
    unknown = {
        p["number"]
        for p in session.gh.pulls()
        if p["state"] == "open" and p["number"] not in latest
    }
    if unknown:
        value = _merge_request(session, enabled, runs[-1], unknown)
        results.update({entry["pr"]: entry for entry in value["results"]})
        evaluations.extend(value["evaluations"])
    for _ in range(len(runs) + 1):
        progressed = False
        for request in runs:
            selected = {
                n
                for n, run_id in latest.items()
                if run_id == request["review_run_id"]
                and results.get(n, {}).get("action") not in {"merged", "already_merged"}
            }
            if not selected:
                continue
            value = _merge_request(session, enabled, request, selected)
            for entry in value["results"]:
                results[entry["pr"]] = entry
                progressed |= entry["action"] == "merged"
            evaluations.extend(value["evaluations"])
        if (
            not enabled
            or not progressed
            or all(
                results.get(n, {}).get("action") in {"merged", "already_merged"}
                for n in latest
            )
        ):
            break
    return {
        "parent": session.parent,
        "review_run_id": runs[-1]["review_run_id"],
        "auto_merge": enabled,
        "agent_invocations": 0,
        "reasons": [] if results else ["no_target_prs"],
        "results": [results[n] for n in sorted(results)],
        "evaluations": evaluations,
    }


def _merge_request(
    session: ParentSession,
    enabled: bool,
    request: dict[str, Any] | None,
    selected: set[int] | None,
) -> dict[str, Any]:
    from decision_logs.review_capture import gate_result

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
            if n in handled or (selected is not None and n not in selected):
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
            gate_result(entry, "evaluation")
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
            gate_result(attempt, "merge_attempt")
            with session.store.unlocked_read():
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
            gate_result(attempt, "merge_result")
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repo")
    root.add_argument("--parent", required=True)
    root.add_argument("--state-dir", type=Path, default=state_directory())
    commands = root.add_subparsers(dest="command", required=True)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--decision-id")
    merge_parser.add_argument("--plan-id")
    merge_parser.add_argument("--retry-of")
    merge_parser.add_argument("--changes-file", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from decision_logs.capture import (
        CURRENT,
        CaptureInputError,
        Operation,
        checked_link,
        comparison,
        initialization_failed,
        observe,
    )
    from decision_logs.journal import Journal
    from decision_logs.parent_commands import failed

    token = None
    try:
        enabled = auto_merge(os.environ.get("AGENT_AUTO_MERGE"))
        identity = session_identity(dict(os.environ))
        store = Store(args.state_dir)
        gh = GitHub(args.repo, store=store)
        parent = issue_number(args.parent, gh.repo)
        try:
            plan = checked_link(
                Journal(store.directory, gh.repo, parent),
                args.decision_id,
                args.plan_id,
                "merge",
            )
            capture = Operation.begin(
                store.directory,
                gh.repo,
                parent,
                "merge_gate.merge",
                "merge",
                decision_id=args.decision_id,
                plan_id=plan,
                previous_result_id=args.retry_of,
                changes=comparison(args.changes_file),
            )
        except CaptureInputError:
            raise
        except Exception:
            initialization_failed(store.directory, gh.repo, parent, "merge_gate.merge")
            capture = None
        token = CURRENT.set(capture)
        with store.locked(), gh.fresh_reads():
            session = ParentSession(
                gh, store, issue_number(args.parent, gh.repo), identity
            )
            value = merge(session, enabled)
        observe(
            value,
            status="success"
            if value["results"]
            and all(r["decision"] == "merge" for r in value["results"])
            else "rejected",
            summary=f"merge_gate.merge: auto_merge={enabled}; results={len(value['results'])}",
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
        failed(exc)
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
    finally:
        if token is not None:
            CURRENT.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
