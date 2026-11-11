#!/usr/bin/env python3
"""Run one integrated Codex review only on an explicit, recorded parent request."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "lib"))

from decision_logs import review_capture as decision_capture  # noqa: E402
from decision_logs.capture import CURRENT  # noqa: E402
from opsproc import CommandError, run_process_group  # noqa: E402
from parent.execution import (  # noqa: E402
    require_requirement_targets,
    require_run_authorization,
)
from parent.github import GitHub  # noqa: E402
from parent.policy import (  # noqa: E402
    closing_issues,
    digest,
    object_value,
    review_state,
    select_prs,
)
from parent.session import ParentSession  # noqa: E402
from parent.snapshot import git, snapshot  # noqa: E402
from parent.state import Store, session_identity, state_directory  # noqa: E402
from parent_review_policy import (  # noqa: E402
    BaseAdvance,
    evaluate_coverage,
    evidence,
    execution_stages,
    hex_value,
    load_json,
    marker,
    pending_handoff_candidates,
    pr_findings,
    records,
    request_manifest,
    superseding_reviews,
    validate_evidence_completion,
    validate_resolution,
    validate_result,
)
from review_interruption import (  # noqa: E402
    CLOSED,
    agent_started,
    cleanup_evidence,
    exit_confirmation_record,
    interruption_record,
    require_not_closed,
    runtime_lock,
    supervisor_confirmation,
)
from review_locks import require_finished_overlaps, target_locks  # noqa: E402
from review_observations import observe  # noqa: E402
from review_policy import neutralize_markers, normalize_rollup  # noqa: E402
from review_recovery import (  # noqa: E402
    file_digest,
    parse_final,
    runtime_evidence,
    source_evidence,
    verify_adoption,
    verify_result_runtime,
)
from review_workspace import (  # noqa: E402
    build_prompt,
    changed_paths,
    full_diff,
    workspace,
)

MAX_COMMENT_CHARS = 60_000


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return object_value(load_json(path.read_text(encoding="utf-8")))


def authorized(
    session: ParentSession,
    request: dict[str, Any],
    *,
    launching: bool = False,
    required_issues: set[int] | None = None,
) -> dict[str, Any]:
    row = session.agreed()
    if not any(run["request"] == request for run in row["reviews"]):
        raise ValueError("review request is unknown, changed or superseded")
    history = session.history()
    state = review_state(history, request)
    if (
        state is None
        or state == "targets_fixed"
        or (launching and state != "instructed")
    ):
        raise ValueError("parent has not explicitly instructed this review")
    instruction = next(
        e
        for e in reversed(history)
        if e.get("event") == "review"
        and e.get("review_run_id") == request["review_run_id"]
        and e.get("state") == "instructed"
    )
    auth = require_run_authorization(session, request, required_issues=required_issues)
    if instruction.get("request_sha256") != digest(request):
        raise ValueError("review instruction refers to another request")
    if request["schema_version"] == 3:
        from parent.review_choice import verify_references

        verify_references(session, request["review_selection"], request["prs"])
    return auth


def collect(
    session: ParentSession, request: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, bytes]]:
    parent, children = session.work_items()
    selected = [p["number"] for p in request["prs"]]
    prs = select_prs(selected, session.gh.pulls(), set(children), session.gh.repo)

    def assignments(pulls: Sequence[Mapping[str, Any]]) -> dict[int, set[int]]:
        return {
            pr["number"]: closing_issues(pr.get("body") or "", session.gh.repo)
            & set(children)
            for pr in pulls
        }

    fixed_assignments = assignments(prs)
    require_requirement_targets(
        request, set().union(*fixed_assignments.values()), exact=True
    )
    actual, diffs, details = [], {}, []
    for pr in prs:
        entry, diff = snapshot(dict(pr))
        actual.append(entry)
        n = pr["number"]
        diffs[n] = diff
        ci = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(n),
                "--repo",
                session.gh.repo,
                "--json",
                "statusCheckRollup",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        rollup = read_rollup(ci.stdout)
        details.append(
            {
                "number": n,
                "title": pr["title"],
                "body": pr["body"],
                "child_issues": sorted(
                    closing_issues(pr["body"] or "", session.gh.repo) & set(children)
                ),
                "changed_paths": changed_paths(
                    entry["merge_base_sha"], entry["head_sha"]
                ),
                "ci": [
                    {"name": c.name, "state": c.state.value}
                    for c in normalize_rollup(rollup)
                ],
            }
        )
    if actual != request["prs"]:
        raise ValueError(
            "head/base/merge-base/diff changed since the instructed snapshot; 対象の縮約／親判断待ち"
        )
    selected_issues = set().union(*fixed_assignments.values())
    observations = observe(
        session,
        [{"number": parent.number, "title": parent.title, "body": parent.body}]
        + [
            {"number": c.number, "title": c.title, "body": c.body}
            for c in children.values()
            if c.number in selected_issues and c.number != parent.number
        ],
        prs,
    )
    # Recollect after all GitHub/git I/O. Never review a partially collected PR set.
    with session.gh.fresh_reads():
        if set(session.work_items()[1]) != set(children):
            raise ValueError("work targets changed during review collection")
        fresh = select_prs(
            selected,
            session.gh.pulls(),
            set(children),
            session.gh.repo,
        )
    if [snapshot(dict(pr))[0] for pr in fresh] != request["prs"] or assignments(
        fresh
    ) != fixed_assignments:
        raise ValueError("PR set/snapshots changed during collection")
    return {
        "parent": {"number": parent.number, "title": parent.title, "body": parent.body},
        "parent_comments": session.gh.comments(parent.number),
        "children": [
            {"number": c.number, "title": c.title, "body": c.body}
            for c in children.values()
            if c.number != parent.number
        ],
        "prs": details,
        "review_observations": observations,
    }, diffs


def read_rollup(text: str) -> list[dict[str, Any]]:
    data = read_object_text(text)
    if set(data) != {"statusCheckRollup"} or not isinstance(
        data["statusCheckRollup"], list
    ):
        raise ValueError("CI collection is missing or incomplete")
    return [object_value(item) for item in data["statusCheckRollup"]]


def read_object_text(text: str) -> dict[str, Any]:
    return object_value(load_json(text))


def result_record(
    request: dict[str, Any], started: str, previous: list[str]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "result",
        "review_run_id": request["review_run_id"],
        "request_sha256": digest(request),
        "request": request,
        "reviewer": "codex",
        "profile": "review",
        **({"runtime_settings": None} if request["schema_version"] == 3 else {}),
        "started_at": started,
        "finished_at": now(),
        "status": "failed",
        "verdict": "incomplete",
        "summary": "未完了。対象の縮約／親判断待ち。",
        "error": None,
        "exception": {"previous_runs": previous, "reason": request["reason"]}
        if previous
        else None,
        "cross_findings": [],
        "prs": [
            {
                "number": p["number"],
                "child_issues": [],
                "coverage": "not_reviewed",
                "summary": "未レビュー",
                "findings": [],
            }
            for p in request["prs"]
        ],
    }


def comment_body(record: dict[str, Any], summary: str) -> str:
    body: str = marker(record) + "\n\n" + neutralize_markers(summary)
    if len(body) > MAX_COMMENT_CHARS:
        raise ValueError(
            "review result exceeds the GitHub comment limit; 対象の縮約／親判断待ち"
        )
    return body


def finding_text(items: list[dict[str, Any]]) -> str:
    return (
        "\n".join(
            f"- {f['id']} [{f['severity']}] {f['path'] or '(横断)'}:{f['line'] or '-'} — {f['rule']}: {f['detail']}"
            for f in items
        )
        or "阻害指摘なし。"
    )


def summaries(result: dict[str, Any]) -> tuple[str, dict[int, str]]:
    text = f"親統合レビュー `{result['review_run_id']}` — {result['status']} / {result['verdict']}\n\nCodex（reviewプロファイル）、{result['started_at']} → {result['finished_at']}\n\n{result['summary']}"
    if result["request"]["schema_version"] == 3:
        choice = result["request"]["review_selection"]
        if choice["kind"] == "run":
            text += f"\n\n選択: {choice['selected']['model']} / {choice['selected']['effort']} ({choice['specified_by']})\n理由: {choice['reason']}"
        if result["runtime_settings"] is not None:
            observed = result["runtime_settings"]["effective"]
            text += f"\n\n実効モデル: {observed['model']['value']} / {observed['effort']['value']}"
    if result["error"]:
        text += "\n\n未完了: " + result["error"]
    if result["exception"]:
        text += (
            "\n\n通常1回のレビュー予算を超える例外。理由: "
            + result["exception"]["reason"]
        )
    if result["kind"] == "adoption":
        text += f"\n\n採用（{result['mode']}）、AI起動0回。元run: {result['source_run']}\n理由: {result['reason']}"
    text += "\n\n横断指摘:\n" + finding_text(result["cross_findings"])
    by_pr = {}
    for pr in result["prs"]:
        n = pr["number"]
        snap = next(s for s in result["request"]["prs"] if s["number"] == n)
        own = (
            f"PR #{n}: {pr['coverage']}。修正担当: {snap['branch']} / 子issue {pr['child_issues']}\n\n{pr['summary']}\n\n"
            + finding_text(pr_findings(result, n))
        )
        for index, wait in enumerate(pr.get("evidence_waits", []), 1):
            own += f"\n\n証拠待ち {index}: {wait['question']}\n必要な証拠: {wait['required_evidence']}"
        by_pr[n] = own
        text += "\n\n" + own
    return text, by_pr


def publish(session: ParentSession, run_dir: Path, result: dict[str, Any]) -> str:
    """Resume a write-ahead publication by reconciling exact public records."""
    from merge_gate_policy import projection

    validate_result(result)
    execution = read_json(run_dir / "execution.json")

    verify_result_runtime(run_dir, result, execution)
    invocations = 0 if result["kind"] == "adoption" else 1
    if (
        execution.get("request_sha256") != result["request_sha256"]
        or type(execution.get("agent_invocations")) is not int
        or execution["agent_invocations"] != invocations
    ):
        raise ValueError("publication execution/request/invocations mismatch")
    summary, by_pr = summaries(result)
    body = comment_body(result, summary)
    session.public(body)
    pending = {
        "schema_version": 1,
        "kind": "pending",
        "review_run_id": result["review_run_id"],
        "request_sha256": result["request_sha256"],
    }

    def matching(
        n: int, kinds: set[str]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        with session.gh.fresh_reads():
            comments = session.gh.comments(n)
        return [
            (c, r)
            for c in comments
            for r in records([c], session.login)
            if r.get("review_run_id") == result["review_run_id"] and r["kind"] in kinds
        ]

    def stage(**updates: Any) -> None:
        execution = read_json(run_dir / "execution.json")
        execution.setdefault("stages", {}).setdefault("publication", {}).update(updates)
        Store.write_json(run_dir / "execution.json", execution)
        decision_capture.result(
            {
                "publication": execution["stages"]["publication"]["status"],
                "stage": execution["stages"]["publication"],
            },
            artifact="execution.json publication stage",
        )

    stage(status="pending", result_sha256=digest(result))
    parents = matching(session.parent, {"pending", "result", "adoption"})
    if len(parents) > 1 or any(r not in (pending, result) for _, r in parents):
        raise ValueError("conflicting public parent record; cannot resume")
    if parents:
        comment, committed = parents[0]
    else:
        comment = session.gh.write(
            "POST",
            session.endpoint + "/comments",
            {
                "body": comment_body(
                    pending,
                    "結果公開中（未完了）。全対象PRへの公開が完了するまで有効なレビュー結果ではありません。",
                )
            },
        )
        committed = pending
    comment_id = comment["id"]
    if type(comment_id) is not int or comment_id < 1:
        raise ValueError("GitHub returned an invalid parent comment ID")
    url = f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{comment_id}"
    stage(parent_comment_id=comment_id, parent_run_url=url, projections=[])
    done = []
    for pr in result["prs"]:
        n = pr["number"]
        record = projection(result, n, url)
        pr_body = comment_body(
            record,
            f"[親runの結果]({url})\n\n" + by_pr[n],
        )
        session.public(pr_body)
        (run_dir / f"pr-{n}-comment.md").write_text(pr_body, encoding="utf-8")
        existing = matching(n, {"pr"})
        if len(existing) > 1 or any(r != record for _, r in existing):
            raise ValueError(f"conflicting PR {n} projection; cannot resume")
        if not existing:
            if committed != pending:
                raise ValueError("committed result has a missing projection")
            session.gh.write(
                "POST",
                f"repos/{session.gh.repo}/issues/{n}/comments",
                {"body": pr_body},
            )
        done.append(n)
        stage(status="projections", projections=done)
    if committed == pending:
        stage(status="result")
        session.gh.write(
            "PATCH",
            f"repos/{session.gh.repo}/issues/comments/{comment_id}",
            {"body": body},
        )
    (run_dir / "parent-comment.md").write_text(body, encoding="utf-8")
    stage(status="published")
    decision_capture.published(url, body)
    return url


def publish_result(
    session: ParentSession, run_dir: Path, result: dict[str, Any]
) -> str:
    """Only publication state changes on failure; the immutable result survives."""
    decision_capture.result({"publication": "pending"}, artifact="publication")
    try:
        url = publish(session, run_dir, result)
        execution = read_json(run_dir / "execution.json")
        execution.update(status="published", parent_run_url=url)
    except (Exception, KeyboardInterrupt) as exc:
        (run_dir / "publication-error.log").write_text(str(exc), encoding="utf-8")
        execution = read_json(run_dir / "execution.json")
        execution.update(status="publication_failed")
        execution.setdefault("stages", {}).setdefault("publication", {}).update(
            status="failed", error=public_error(session, exc)
        )
    execution["finished_at"] = now()
    Store.write_json(run_dir / "execution.json", execution)
    decision_capture.result(
        {"publication": execution["status"]}, artifact="publication"
    )
    return str(execution["status"])


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    request = request_manifest(read_json(args.request))
    capture = decision_capture.initialize(
        args, request, "review.run", lambda: decision_capture.begin(args, request)
    )
    token = CURRENT.set(capture)
    try:
        value = _locked_review(args, request)
        decision_capture.complete(value)
        return value
    except BaseException as exc:
        from decision_logs.parent_commands import failed

        failed(exc)
        raise
    finally:
        CURRENT.reset(token)


def _locked_review(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    directory = args.state_dir / "reviews" / request["review_run_id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (
        target_locks(args.state_dir, request),
        (directory / "run.lock").open("a") as lock,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this review run is already active") from exc
        return _run_review(args)


def _run_review(args: argparse.Namespace) -> dict[str, Any]:
    request = request_manifest(read_json(args.request))
    from lib.review_settings import (
        LEGACY_FIXED,
        load_policy,
        runtime_summary,
        verify_execution,
    )

    choice = request.get("review_selection")
    if choice is not None and choice["kind"] != "run":
        raise ValueError("recovery request cannot launch an agent; use recover")
    gh = GitHub(request["repo"])
    identity = session_identity(dict(os.environ))
    store = Store(args.state_dir)
    run_dir = store.directory / "reviews" / request["review_run_id"]
    with store.locked():
        session = ParentSession(gh, store, request["parent_issue"], identity)
        auth = authorized(session, request)
        require_finished_overlaps(
            store.directory,
            [r for parent_row in store.parents.values() for r in parent_row["reviews"]],
            request,
        )
        if read_json(run_dir / "request.json") != request:
            raise ValueError("local request snapshot does not match")
        execution_file = run_dir / "execution.json"
        if execution_file.exists():
            execution = read_json(execution_file)
            verify_execution(request, execution)
            require_not_closed(execution, request)
            execution_stages(execution)
            if not isinstance(execution.get("status"), str) or execution[
                "status"
            ] not in {
                "running",
                "preparation_failed",
                "published",
                "publication_failed",
            }:
                raise ValueError("unknown review execution status")
            if execution["request_sha256"] != digest(request):
                raise ValueError("run ID was already used with a different snapshot")
            if type(execution.get("agent_invocations")) is not int or execution[
                "agent_invocations"
            ] not in {0, 1}:
                raise ValueError("invalid recorded agent invocation count")
            if execution.get("kind") == "adoption":
                raise ValueError(
                    "adoption cannot launch an agent; use resume-publication"
                )
            if execution.get("agent_invocations") == 0:
                if execution.get("status") != "preparation_failed":
                    raise ValueError("unconfirmed preparation cannot be retried")
                # No reviewer was launched. Preserve the failed preparation and
                # accept the parent's explicit resubmission of this same run.
                attempt = run_dir / "preparation-attempts" / uuid.uuid4().hex
                attempt.mkdir(parents=True, mode=0o700)
                runtime = run_dir / "agent-runtime"
                if runtime.exists():
                    runtime.rename(attempt / "agent-runtime")
                for name in ("execution.json", "result.json", "context.json"):
                    source = run_dir / name
                    if source.exists():
                        (attempt / name).write_bytes(source.read_bytes())
            else:
                if execution["status"] == "running":
                    raise ValueError(
                        "run is running/interrupted after agent launch; never execute it twice"
                    )
                collect(session, request)
                recorded_result = validate_result(read_json(run_dir / "result.json"))

                verify_result_runtime(run_dir, recorded_result, execution)
                return {
                    "reused": True,
                    "result": recorded_result,
                    "publication": execution["status"],
                }
        authorized(session, request, launching=True)
        if choice is not None:
            from lib.agent_settings import DEFAULT_CONFIG, read_pins
            from lib.review_settings import launch_selection

            launch_selection(
                choice, read_pins(args.review_config_file or DEFAULT_CONFIG)
            )
        else:
            legacy_policy = load_policy(args.review_config_file)
            if (
                legacy_policy["mode"] != "fixed"
                or legacy_policy["fixed"] != LEGACY_FIXED
            ):
                raise ValueError(
                    "legacy request requires original fixed settings; prepare a new request"
                )
        for pr in request["prs"]:
            if (
                hashlib.sha256(
                    (run_dir / f"pr-{pr['number']}.diff").read_bytes()
                ).hexdigest()
                != pr["diff_sha256"]
            ):
                raise ValueError("local frozen diff digest mismatch")
        execution = {
            "request_sha256": digest(request),
            **({"review_selection": choice} if choice is not None else {}),
            "status": "running",
            "started_at": now(),
            "authorization": request.get("authorization", auth),
            "operator_authorization": auth,
            "session": identity,
            "agent_invocations": 0,
            "stage_version": 1,
            "stages": {
                "agent": {"status": "not_started", "cleanup_complete": False},
                "output": {"status": "not_started"},
                "postcheck": {"status": "not_started"},
                "publication": {"status": "not_started"},
            },
        }
        Store.write_json(execution_file, execution)
        previous = []
        for prior in session.row["reviews"][:-1]:
            prior_id = prior["request"]["review_run_id"]
            prior_execution = store.directory / "reviews" / prior_id / "execution.json"
            if (
                prior_execution.exists()
                and read_json(prior_execution)["agent_invocations"]
            ):
                previous.append(prior_id)
    error: BaseException | None = None
    try:
        context, diffs = collect(session, request)
        context["instruction"] = next(
            run.get("evidence", "")
            for run in session.row["reviews"]
            if run["request"] == request
        )
        if choice is not None and choice["handoff"] is not None:
            from parent.review_choice import source

            source_request, source_result = source(
                session, choice["handoff"]["source_run"]
            )
            context["review_handoff"] = {
                "source_request": source_request,
                "source_result": source_result,
                "scope": choice["handoff"]["scope"],
                "candidates": choice["handoff"]["candidates"],
            }
        Store.write_json(run_dir / "context.json", context)
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        template = (HERE / "prompt_template.md").read_text(encoding="utf-8")
        prompt = build_prompt(request, context, diffs, template)
        prompt_file, output_file = run_dir / "prompt.md", run_dir / "agent_output.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        if choice is not None:
            Store.write_json(run_dir / "review-selection.json", choice)
        cpu_tests = read_json(args.cpu_tests_file) if args.cpu_tests_file else {}
        retrieval_config = {
            "repo": str(root),
            "repository": request["repo"],
            "run_id": request["review_run_id"],
            "commits": sorted(
                {
                    pr[key]
                    for pr in request["prs"]
                    for key in ("head_sha", "base_sha", "merge_base_sha")
                }
            ),
            "journal_dir": str(run_dir / "retrieval"),
            "cpu_tests": cpu_tests,
            "isolation_proof": str(args.isolation_proof)
            if args.isolation_proof
            else None,
        }
        Store.write_json(run_dir / "retrieval-config.json", retrieval_config)
        mcp_config = run_dir / "mcp-config.json"
        Store.write_json(
            mcp_config,
            {
                "review": {
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "lib.review_tools.review_server",
                        "--config",
                        str(run_dir / "retrieval-config.json"),
                    ],
                    "env": {
                        "PYTHONPATH": str(HERE.parent),
                        **decision_capture.retrieval_env(),
                    },
                    "tool_timeout_sec": 960,
                }
            },
        )

        # Every accepted run reaches this call at most once. No per-PR agent loop.
        with workspace(root, request, context, diffs) as cwd:
            with store.locked():
                if authorized(session, request) != auth:
                    raise ValueError("parent authorization changed before review")
                execution["agent_invocations"] = 1
                execution["stages"]["agent"]["status"] = "running"
                Store.write_json(execution_file, execution)
            run_process_group(
                [
                    str(args.agent_exec),
                    "--agent",
                    "codex",
                    "--profile",
                    "review",
                    *(
                        [
                            "--review-selection-file",
                            str(run_dir / "review-selection.json"),
                        ]
                        if choice is not None
                        else []
                    ),
                    *(
                        ["--config-file", str(args.review_config_file.resolve())]
                        if args.review_config_file is not None
                        else []
                    ),
                    "--cwd",
                    str(cwd),
                    "--prompt-file",
                    str(prompt_file),
                    "--out",
                    str(output_file),
                    "--mcp-config",
                    str(mcp_config),
                    "--json-schema",
                    str(HERE / "result_schema.json"),
                    "--mode",
                    "read-only",
                    "--log-dir",
                    str(run_dir),
                    "--run-dir",
                    str(run_dir / "agent-runtime"),
                ],
                cwd=root,
                timeout=args.agent_timeout,
                env=decision_capture.runtime_env(run_dir / "agent-runtime"),
            )
    except (Exception, KeyboardInterrupt) as exc:
        (run_dir / "error.log").write_text(str(exc), encoding="utf-8")
        error = exc
    # Parse and preserve final independently of the runtime and postcheck outcomes.
    candidate = None
    if execution["agent_invocations"]:
        agent = execution["stages"]["agent"]
        review_workspace = (
            root / ".claude/worktrees" / ("review-" + request["review_run_id"])
        )
        agent["workspace_cleanup_complete"] = not review_workspace.exists()
        try:
            runtime = runtime_evidence(run_dir, request)
            agent.update(
                status="finished",
                cleanup_complete=True,
                runtime_sha256=digest(runtime),
                exit_code=runtime["exit_code"],
            )
            if runtime["result"] != "completed" or runtime["exit_code"] != 0:
                raise ValueError(
                    "review runtime failed; saved final requires explicit recovery"
                )
            if not agent["workspace_cleanup_complete"]:
                raise ValueError("review workspace cleanup is incomplete")
        except (Exception, KeyboardInterrupt) as exc:
            if agent["status"] != "finished":
                agent["status"] = "unconfirmed"
            agent["error"] = public_error(session, exc)
            error = error or exc
        try:
            final, parsed = parse_final(run_dir, request)
            candidate = result_record(request, execution["started_at"], previous)
            if choice is not None:
                candidate["runtime_settings"] = runtime_summary(
                    read_json(run_dir / "agent-runtime/result.json")
                )
            candidate.update(parsed)
            for pr in candidate["prs"]:
                pr["child_issues"] = next(
                    p["child_issues"]
                    for p in context["prs"]
                    if p["number"] == pr["number"]
                )
            candidate["status"] = (
                "succeeded" if parsed["verdict"] != "incomplete" else "failed"
            )
            candidate["error"] = (
                None
                if candidate["status"] == "succeeded"
                else "未確認の対象があります。"
            )
            validate_result(candidate)
            Store.write_json(run_dir / "output.json", candidate)
            decision_capture.result(candidate, artifact="output.json")
            execution["stages"]["output"] = {
                "status": "validated",
                "final_file": str(final.relative_to(run_dir)),
                "final_sha256": file_digest(final),
                "output_sha256": digest(candidate),
            }
        except (Exception, KeyboardInterrupt) as exc:
            execution["stages"]["output"] = {
                "status": "invalid",
                "error": public_error(session, exc),
            }
            candidate = None
            error = error or exc
        Store.write_json(execution_file, execution)
        try:
            with store.locked():
                if authorized(session, request) != auth:
                    raise ValueError("parent authorization changed during review")
                with gh.fresh_reads():
                    fresh_context, _ = collect(session, request)
                if any(
                    a["child_issues"] != b["child_issues"]
                    for a, b in zip(context["prs"], fresh_context["prs"], strict=True)
                ):
                    raise ValueError("PR child assignments changed during review")
            execution["stages"]["postcheck"] = {
                "status": "passed",
                "snapshots_sha256": digest(request["prs"]),
            }
        except (Exception, KeyboardInterrupt) as exc:
            execution["stages"]["postcheck"] = {
                "status": "failed",
                "error": public_error(session, exc),
            }
            error = error or exc
    runtime_result = run_dir / "agent-runtime" / "result.json"
    if runtime_result.exists():
        try:
            runtime_record = read_json(runtime_result)
            if (
                runtime_record.get("result") == "failed"
                and runtime_record.get("agent_spawned") is False
                and runtime_record.get("cleanup_complete") is True
            ):
                execution["agent_invocations"] = 0
        except (Exception, KeyboardInterrupt) as exc:
            error = error or exc
    if execution["agent_invocations"] == 0 and not agent_started(execution):
        agent = execution["stages"]["agent"]
        agent["cleanup_complete"] = True
        try:
            cleanup_evidence(run_dir, request, execution)
            agent["workspace_cleanup_complete"] = True
        except (Exception, KeyboardInterrupt) as exc:
            agent["workspace_cleanup_complete"] = False
            error = error or exc
    result = (
        candidate
        if candidate is not None and error is None
        else result_record(request, execution["started_at"], previous)
    )
    if error is not None:
        result["error"] = public_error(session, error)
    elif candidate is None:
        result["error"] = "レビューを開始できませんでした。"
    # output.json is immutable source evidence, including its timestamps.
    if result is not candidate:
        result["finished_at"] = now()
    validate_result(result)
    Store.write_json(run_dir / "result.json", result)
    Store.write_json(execution_file, execution)
    decision_capture.result(result, artifact="result.json")
    if execution["agent_invocations"] == 0:
        execution.update(status="preparation_failed", finished_at=now())
        Store.write_json(execution_file, execution)
        return {"reused": False, "result": result, "publication": "preparation_failed"}
    publication = publish_result(session, run_dir, result)
    return {"reused": False, "result": result, "publication": publication}


def public_error(session: ParentSession, exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for row in session.store.parents.values():
        text = text.replace(row["session"].partition(":")[2], "[session]")
        for task in row["tasks"].values():
            if task.get("worktree"):
                text = text.replace(task["worktree"], "[worktree]")
    for root in (session.store.directory, Path.cwd(), HERE.parents[2]):
        text = text.replace(str(root), "[local]")
    return text[:1500]


def recover(args: argparse.Namespace) -> dict[str, Any]:
    request = request_manifest(read_json(args.request))
    if request["schema_version"] == 3 and (
        request["review_selection"]["kind"] != "recover"
        or request["review_selection"]["source_run"] != args.source_run
    ):
        raise ValueError("recovery source differs from frozen request")
    if (
        not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.source_run)
        or args.source_run == request["review_run_id"]
    ):
        raise ValueError("recovery requires a distinct valid source run")
    reason = args.reason_file.read_text(encoding="utf-8").strip()
    if not reason:
        raise ValueError("recovery requires the parent's reason")
    store = Store(args.state_dir)
    run_dir = store.directory / "reviews" / request["review_run_id"]
    with target_locks(store.directory, request), ExitStack() as stack:
        # Fixed ordering prevents concurrent recoveries from deadlocking.
        for run in sorted((request["review_run_id"], args.source_run)):
            lock = stack.enter_context(
                (store.directory / "reviews" / run / "run.lock").open("a")
            )
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("review run is active; cannot recover") from exc
        with store.locked():
            session = ParentSession(
                GitHub(request["repo"]),
                store,
                request["parent_issue"],
                session_identity(dict(os.environ)),
            )
            auth = authorized(session, request)
            require_finished_overlaps(
                store.directory,
                [
                    r
                    for parent_row in store.parents.values()
                    for r in parent_row["reviews"]
                ],
                request,
            )
            if read_json(run_dir / "request.json") != request:
                raise ValueError("local recovery request mismatch")
            if (run_dir / "execution.json").exists():
                require_not_closed(read_json(run_dir / "execution.json"), request)
            candidate, validation = source_evidence(store.directory, args.source_run)
            source = candidate["request"]
            authorized(session, source)
            if superseding_reviews(session.row["reviews"], args.source_run, request):
                raise ValueError(
                    "source review is superseded; use the latest relevant execution"
                )
            if any(request[k] != source[k] for k in ("repo", "parent_issue", "prs")):
                raise ValueError("recover requires unchanged source targets")
            with session.gh.fresh_reads():
                current_context, _ = collect(session, request)
            if any(
                a["child_issues"] != b["child_issues"]
                for a, b in zip(candidate["prs"], current_context["prs"], strict=True)
            ):
                raise ValueError("source PR child assignments changed")
            execution_path = run_dir / "execution.json"
            if execution_path.exists():
                result = validate_result(read_json(run_dir / "result.json"))
                if (
                    result["kind"] != "adoption"
                    or result["source_run"] != args.source_run
                    or result["reason"] != reason
                    or result["request"] != request
                ):
                    raise ValueError("recovery run already contains different evidence")
                verify_adoption(store.directory, result)
            else:
                result = {
                    **candidate,
                    "kind": "adoption",
                    "review_run_id": request["review_run_id"],
                    "request": request,
                    "request_sha256": digest(request),
                    "started_at": now(),
                    "finished_at": now(),
                    "exception": None,
                    "mode": "recover",
                    "source_run": args.source_run,
                    "reason": reason,
                    "agent_invocations": 0,
                    "validation": validation,
                }
                if request["schema_version"] == 3:
                    from lib.review_settings import runtime_summary

                    result["runtime_settings"] = runtime_summary(
                        read_json(
                            store.directory
                            / "reviews"
                            / args.source_run
                            / "agent-runtime/result.json"
                        )
                    )
                verify_adoption(store.directory, result)
                for pr in request["prs"]:
                    if (
                        file_digest(run_dir / f"pr-{pr['number']}.diff")
                        != pr["diff_sha256"]
                    ):
                        raise ValueError("recovery frozen diff mismatch")
                Store.write_json(run_dir / "result.json", result)
                Store.write_json(
                    execution_path,
                    {
                        "kind": "adoption",
                        "stage_version": 1,
                        "agent_invocations": 0,
                        "source_run": args.source_run,
                        "request_sha256": digest(request),
                        **(
                            {"review_selection": request["review_selection"]}
                            if request["schema_version"] == 3
                            else {}
                        ),
                        "authorization": request.get("authorization", auth),
                        "operator_authorization": auth,
                        "session": session.session,
                        "status": "running",
                        "started_at": result["started_at"],
                        "stages": {
                            "agent": {
                                "status": "not_invoked",
                                "cleanup_complete": True,
                                "workspace_cleanup_complete": True,
                            },
                            "output": {
                                "status": "validated",
                                "output_sha256": validation["output_sha256"],
                            },
                            "postcheck": {
                                "status": "passed",
                                "snapshots_sha256": digest(request["prs"]),
                            },
                            "publication": {"status": "not_started"},
                        },
                    },
                )
        publication = publish_result(session, run_dir, result)
    return {"result": result, "publication": publication, "agent_invocations": 0}


def close_interrupted(args: argparse.Namespace) -> dict[str, Any]:
    """Confirm exit explicitly; close only running runs, preserve formal results."""
    if not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.run):
        raise ValueError("invalid run ID")
    reason = args.reason_file.read_text(encoding="utf-8").strip()
    if not reason:
        raise ValueError("interrupted closure requires the parent's reason")
    store = Store(args.state_dir)
    run_dir = store.directory / "reviews" / args.run
    request = request_manifest(read_json(run_dir / "request.json"), instructed=False)
    if request["review_run_id"] != args.run:
        raise ValueError("interrupted run/request mismatch")
    with (
        target_locks(store.directory, request),
        (run_dir / "run.lock").open("a") as lock,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("review run is active; cannot close interruption") from exc
        with runtime_lock(run_dir), store.locked():
            session = ParentSession(
                GitHub(request["repo"]),
                store,
                request["parent_issue"],
                session_identity(dict(os.environ)),
            )
            auth = authorized(session, request)
            if read_json(run_dir / "request.json") != request:
                raise ValueError("local interrupted request changed")
            path = run_dir / "execution.json"
            execution = read_json(path)
            terminal = execution.get("status") in {"running", CLOSED}
            key = "interruption" if terminal else "exit_confirmation"
            record = interruption_record(execution, request)
            if not terminal:
                record = exit_confirmation_record(execution, request)
            if record is not None:
                if record["reason"] != reason:
                    raise ValueError(
                        "interrupted closure already has a different reason"
                    )
                if args.cleanup_confirmation is not None and supervisor_confirmation(
                    read_json(args.cleanup_confirmation), request
                ) != record.get("cleanup_confirmation"):
                    raise ValueError(
                        "cleanup confirmation differs from the saved evidence"
                    )
            else:
                if (
                    execution.get("status")
                    not in {
                        "running",
                        "published",
                        "publication_failed",
                        "preparation_failed",
                    }
                    or execution.get("request_sha256") != digest(request)
                    or type(execution.get("agent_invocations")) is not int
                    or execution["agent_invocations"] not in {0, 1}
                ):
                    raise ValueError(
                        "only an unchanged recorded execution can be confirmed"
                    )
                runtime_sha256, supervisor = cleanup_evidence(
                    run_dir, request, execution, args.cleanup_confirmation
                )
                record = {
                    "schema_version": 1,
                    "request_sha256": digest(request),
                    "closed_at" if terminal else "confirmed_at": now(),
                    "reason": reason,
                    "operator_authorization": auth,
                    "session": session.session,
                    "agent_started": agent_started(execution),
                    "runtime_sha256": runtime_sha256,
                    "cleanup_confirmation": supervisor,
                    "workspace_cleanup_complete": True,
                    "publication": "not_published" if terminal else "unchanged",
                }
                if terminal:
                    execution.update(
                        status=CLOSED,
                        finished_at=record["closed_at"],
                        interruption=record,
                    )
                    interruption_record(execution, request)
                else:
                    execution["exit_confirmation"] = record
                    exit_confirmation_record(execution, request)
                Store.write_json(path, execution)
    return {
        "review_run_id": args.run,
        "status": execution["status"],
        "publication": record["publication"],
        "agent_invocations": 0,
        key: record,
    }


def resume_publication(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.run):
        raise ValueError("invalid run ID")
    store = Store(args.state_dir)
    run_dir = store.directory / "reviews" / args.run
    request = request_manifest(read_json(run_dir / "request.json"), instructed=False)
    with (
        target_locks(store.directory, request),
        (run_dir / "run.lock").open("a") as lock,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("review run is active; cannot resume publication") from exc
        require_not_closed(read_json(run_dir / "execution.json"), request)
        result = validate_result(read_json(run_dir / "result.json"))
        request = result["request"]
        with store.locked():
            session = ParentSession(
                GitHub(request["repo"]),
                store,
                request["parent_issue"],
                session_identity(dict(os.environ)),
            )
            authorized(session, request)
            require_finished_overlaps(
                store.directory,
                [
                    r
                    for parent_row in store.parents.values()
                    for r in parent_row["reviews"]
                ],
                request,
            )
            execution = read_json(run_dir / "execution.json")

            verify_result_runtime(run_dir, result, execution)
            if execution["status"] == "running":
                stages = execution_stages(execution)
                if (
                    stages is None
                    or stages["postcheck"].get("status") != "passed"
                    or stages["output"].get("status") != "validated"
                ):
                    raise ValueError("interrupted run has unconfirmed output/postcheck")
                if result["kind"] == "result":
                    runtime_evidence(run_dir)
                    if (
                        stages["agent"].get("workspace_cleanup_complete") is not True
                        or read_json(run_dir / "output.json") != result
                    ):
                        raise ValueError("interrupted run is not ready for publication")
            if (
                execution["status"]
                not in {"published", "publication_failed", "running"}
                or execution["request_sha256"] != digest(request)
                or read_json(run_dir / "request.json") != request
            ):
                raise ValueError("run is not ready for publication resume")
            if result["kind"] == "adoption":
                verify_adoption(store.directory, result)
                if superseding_reviews(
                    session.row["reviews"], result["source_run"], request
                ):
                    raise ValueError(
                        "source review is superseded; cannot resume adoption publication"
                    )
            with session.gh.fresh_reads():
                current_context, _ = collect(session, request)
            if any(p["coverage"] == "reviewed" for p in result["prs"]) and any(
                a["child_issues"] != b["child_issues"]
                for a, b in zip(result["prs"], current_context["prs"], strict=True)
            ):
                raise ValueError("publication PR child assignments changed")
        publication = publish_result(session, run_dir, result)
    return {"result": result, "publication": publication, "agent_invocations": 0}


def complete_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.run):
        raise ValueError("invalid run ID")
    store = Store(args.state_dir)
    run_dir = store.directory / "reviews" / args.run
    result = validate_result(read_json(run_dir / "result.json"))
    request = result["request"]
    with store.locked():
        session = ParentSession(
            GitHub(request["repo"]),
            store,
            request["parent_issue"],
            session_identity(dict(os.environ)),
        )
        entry = next((p for p in result["prs"] if p["number"] == args.pr), None)
        if entry is None or not 1 <= args.wait <= len(entry.get("evidence_waits", [])):
            raise ValueError("unknown evidence wait for this PR")
        authorized(session, request, required_issues=set(entry["child_issues"]))
        execution = read_json(run_dir / "execution.json")
        if execution["status"] != "published" or entry["coverage"] != "reviewed":
            raise ValueError("evidence requires a published code review")
        verify_result_runtime(run_dir, result, execution)
        raw = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{args.pr}"))
        select_prs([args.pr], [raw], set(entry["child_issues"]), session.gh.repo)
        current, _ = snapshot(raw)
        original = next(p for p in request["prs"] if p["number"] == args.pr)
        if current["branch"] != original["branch"]:
            raise ValueError("evidence PR branch changed")
        proof = evidence(read_json(args.evidence))
        if proof["reviewed_paths"] != changed_paths(
            current["merge_base_sha"], current["head_sha"]
        ):
            raise ValueError("evidence reviewed_paths must cover the current PR diff")
        record = {
            "schema_version": 1,
            "kind": "evidence",
            "review_run_id": args.run,
            "result_sha256": digest(result),
            "parent_issue": session.parent,
            "pr": args.pr,
            "wait": args.wait,
            "snapshot": current,
            "evidence": proof,
        }
        validate_evidence_completion(record)
        path = run_dir / f"evidence-{digest(record)}.json"
        saved = (
            read_json(path) if path.exists() else {"record": record, "published": False}
        )
        if saved["record"] != record:
            raise ValueError("evidence digest collision")
        Store.write_json(path, saved)
        body = comment_body(
            record,
            f"PR #{args.pr} 証拠待ち {args.wait} の確認完了（AI起動0回）。\n\nCI: {proof['ci']}\nテスト: {proof['regression_tests']}\n確認: {proof['resolution']}\n確認パス: {proof['reviewed_paths']}",
        )
        session.public(body)
        for n in (args.pr, session.parent):
            with session.gh.fresh_reads():
                comments = session.gh.comments(n)
            if not any(record in records([c], session.login) for c in comments):
                session.gh.write(
                    "POST",
                    f"repos/{session.gh.repo}/issues/{n}/comments",
                    {"body": body},
                )
        saved["published"] = True
        Store.write_json(path, saved)
    return {"evidence": record, "agent_invocations": 0}


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run or not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.run):
        raise ValueError("invalid run ID")
    raw = validate_result(
        read_json(args.state_dir / "reviews" / args.run / "result.json")
    )
    request = request_manifest(raw["request"], instructed=False)
    capture = decision_capture.initialize(
        args,
        request,
        "review.resolve",
        lambda: decision_capture.begin_verification(args, request),
    )
    token = CURRENT.set(capture)
    try:
        value = _resolve(args)
        if capture is not None:
            from decision_logs.capture import observe

            observe(
                value,
                observer="parent_verification",
                verifies=capture.row["verification_targets"],
                summary=f"review.resolve: explicit verification recorded for PR {args.pr}, finding {args.finding}; coverage={value['coverage']}",
            )
        return value
    except BaseException as exc:
        from decision_logs.parent_commands import failed

        failed(exc)
        raise
    finally:
        CURRENT.reset(token)


def _resolve(args: argparse.Namespace) -> dict[str, Any]:
    store = Store(args.state_dir)
    if not args.run or not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.run):
        raise ValueError("invalid run ID")
    run_dir = store.directory / "reviews" / args.run
    result = validate_result(read_json(run_dir / "result.json"))
    request = request_manifest(result["request"], instructed=False)
    with store.locked():
        session = ParentSession(
            GitHub(request["repo"]),
            store,
            request["parent_issue"],
            session_identity(dict(os.environ)),
        )
        child_issues = next(
            (set(p["child_issues"]) for p in result["prs"] if p["number"] == args.pr),
            set(),
        )
        authorized(session, request, required_issues=child_issues)
        execution = read_json(run_dir / "execution.json")
        originals = {p["number"]: p for p in request["prs"]}
        if args.pr not in originals:
            raise ValueError("PR is outside the reviewed set")
        reviewed = next(p for p in result["prs"] if p["number"] == args.pr)
        if execution["status"] != "published" or reviewed["coverage"] != "reviewed":
            raise ValueError(
                "cannot resolve a failed/unpublished review without reviewed PR coverage"
            )

        verify_result_runtime(run_dir, result, execution)
        if args.finding not in {
            "changes",
            *(
                {"handoff-candidates"}
                if pending_handoff_candidates(result, args.pr)
                else set()
            ),
            *(f["id"] for f in pr_findings(result, args.pr)),
        }:
            raise ValueError("unknown finding for this PR")
        fixed = hex_value(args.fixed_head, 40)
        raw = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{args.pr}"))
        # Resolve may follow other PRs being merged, so do not demand the original open set.
        select_prs(
            [args.pr],
            [raw],
            {c for p in result["prs"] for c in p["child_issues"]},
            session.gh.repo,
        )
        current, _ = snapshot(raw)
        if current["head_sha"] != fixed:
            raise ValueError("fixed-head is not the current PR head")
        original = originals[args.pr]
        if original["branch"] != current["branch"]:
            raise ValueError("PR branch identity changed")
        advances: tuple[BaseAdvance, ...] = ()
        diff = full_diff(original["head_sha"], fixed)
        paths = changed_paths(original["head_sha"], fixed)
        proof = evidence(read_json(args.evidence))
        if proof["reviewed_paths"] != paths:
            raise ValueError("actual fix paths differ from the parent's reviewed paths")
        record = {
            "schema_version": 1,
            "kind": "resolution",
            "review_run_id": args.run,
            "result_sha256": digest(result),
            "parent_issue": request["parent_issue"],
            "pr": args.pr,
            "finding": args.finding,
            "reviewed_head": original["head_sha"],
            "fixed_snapshot": current,
            "fix_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "changed_paths": paths,
            "evidence": proof,
        }
        validate_resolution(record)
        key = digest(record)
        record_path = run_dir / f"resolution-{key}.json"
        if record_path.exists():
            saved = read_json(record_path)
            if saved["record"] != record:
                raise ValueError("resolution digest collision")
        else:
            saved = {"record": record, "published": False, "recorded_at": now()}
            Store.write_json(record_path, saved)
            (run_dir / f"resolution-{key}.diff").write_bytes(diff)
        if not saved["published"]:
            text = f"指摘 {args.finding} / PR #{args.pr} の修正確認\n\n[親run]({execution['parent_run_url']})\n\n修正head: {fixed}\n差分SHA-256: {record['fix_diff_sha256']}\n変更ファイル: {paths}\n\nCI: {proof['ci']}\n回帰テスト: {proof['regression_tests']}\n解消根拠: {proof['resolution']}\n\nCodex再実行なし。"
            body = comment_body(record, text)
            session.public(body)
            # Parent record is committed only after the PR projection is posted.
            for n in (args.pr, session.parent):
                endpoint = f"repos/{session.gh.repo}/issues/{n}/comments"
                if not any(
                    c.get("body") == body
                    and c.get("user", {}).get("login") == session.login
                    for c in session.gh.comments(n)
                ):
                    session.gh.write("POST", endpoint, {"body": body})
            saved["published"] = True
            Store.write_json(record_path, saved)
        resolved = [
            read_json(p)["record"]
            for p in run_dir.glob("resolution-*.json")
            if read_json(p)["published"]
        ]
        coverage = evaluate_coverage(
            result,
            [current if p["number"] == args.pr else p for p in request["prs"]],
            resolved,
            {args.pr: paths},
            fix_diff_digests={args.pr: record["fix_diff_sha256"]},
            base_advances=advances,
        )
        return {
            "resolution": record,
            "coverage": {
                "covered": coverage.covered,
                "reasons": coverage.reasons,
                "unresolved": coverage.unresolved,
            },
        }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--state-dir",
        type=Path,
        default=state_directory(),
        help="same private state as parent CLI",
    )
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument(
        "--cpu-tests-file",
        type=Path,
        help="parent-approved mapping of pinned SHA to existing CPU test nodeids",
    )
    run.add_argument("--isolation-proof", type=Path)
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--review-config-file", type=Path)
    run.add_argument(
        "--agent-exec", type=Path, default=HERE.parent / "lib/agent_exec.sh"
    )
    run.add_argument("--agent-timeout", type=float, default=1800)
    run.add_argument("--decision-id")
    run.add_argument("--plan-id")
    run.add_argument("--retry-of")
    run.add_argument("--changes-file", type=Path)
    fix = commands.add_parser("resolve")
    fix.add_argument("--run", required=True)
    fix.add_argument("--pr", type=int, required=True)
    fix.add_argument("--finding", required=True)
    fix.add_argument("--fixed-head", required=True)
    fix.add_argument("--evidence", type=Path, required=True)
    fix.add_argument("--decision-id")
    fix.add_argument("--plan-id")
    fix.add_argument("--retry-of")
    fix.add_argument("--changes-file", type=Path)
    recovery = commands.add_parser("recover")
    recovery.add_argument("--request", type=Path, required=True)
    recovery.add_argument("--source-run", required=True)
    recovery.add_argument("--reason-file", type=Path, required=True)
    resume = commands.add_parser("resume-publication")
    resume.add_argument("--run", required=True)
    close = commands.add_parser("close-interrupted")
    close.add_argument("--run", required=True)
    close.add_argument("--reason-file", type=Path, required=True)
    close.add_argument("--cleanup-confirmation", type=Path)
    complete = commands.add_parser("complete-evidence")
    complete.add_argument("--run", required=True)
    complete.add_argument("--pr", type=int, required=True)
    complete.add_argument("--wait", type=int, required=True)
    complete.add_argument("--evidence", type=Path, required=True)
    for control in (recovery, resume, complete):
        control.add_argument("--decision-id")
        control.add_argument("--plan-id")
        control.add_argument("--retry-of")
        control.add_argument("--changes-file", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run" and (
            args.agent_timeout <= 0 or not math.isfinite(args.agent_timeout)
        ):
            raise ValueError("agent-timeout must be positive and finite")
        operation = {
            "run": run_review,
            "resolve": resolve,
            "recover": recover,
            "resume-publication": resume_publication,
            "close-interrupted": close_interrupted,
            "complete-evidence": complete_evidence,
        }[args.command]
        if args.command in {"recover", "resume-publication", "complete-evidence"}:
            if args.command == "recover":
                request = request_manifest(read_json(args.request))
            else:
                if not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", args.run):
                    raise ValueError("invalid run ID")
                run_dir = args.state_dir / "reviews" / args.run
                if (run_dir / "execution.json").exists():
                    # A closed run never has a formal result; say so before reading one.
                    require_not_closed(
                        read_json(run_dir / "execution.json"),
                        request_manifest(
                            read_json(run_dir / "request.json"), instructed=False
                        ),
                    )
                request = validate_result(
                    read_json(args.state_dir / "reviews" / args.run / "result.json")
                )["request"]
            value = decision_capture.control(args, request, operation)
        else:
            value = operation(args)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if args.command in {"run", "recover", "resume-publication"}:
            return (
                0
                if value["result"]["status"] == "succeeded"
                and value["publication"] == "published"
                else 1
            )
        return 0
    except (
        ValueError,
        OSError,
        CommandError,
        subprocess.SubprocessError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"parent_review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
