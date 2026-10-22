#!/usr/bin/env python3
"""Run one integrated Codex review only on an explicit, recorded parent request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "lib"))

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
    hex_value,
    load_json,
    marker,
    parse_output,
    pr_findings,
    request_manifest,
    validate_resolution,
    validate_result,
)
from review_policy import neutralize_markers, normalize_rollup  # noqa: E402
from review_workspace import (  # noqa: E402
    build_prompt,
    changed_paths,
    full_diff,
    verify_base_advances,
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
    if not row["reviews"] or row["reviews"][-1]["request"] != request:
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
    if any(instruction.get(k) != v for k, v in auth.items()):
        raise ValueError("review instruction belongs to an old owner/agreement")
    return auth


def collect(
    session: ParentSession, request: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, bytes]]:
    parent, children = session.family()
    selected = [p["number"] for p in request["prs"]]
    prs = select_prs(
        selected, session.gh.pulls(), set(children) or {parent.number}, session.gh.repo
    )

    def assignments(pulls: Sequence[Mapping[str, Any]]) -> dict[int, set[int]]:
        return {
            pr["number"]: closing_issues(pr.get("body") or "", session.gh.repo)
            & (set(children) or {parent.number})
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
                    closing_issues(pr["body"] or "", session.gh.repo)
                    & (set(children) or {parent.number})
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
    # Recollect after all GitHub/git I/O. Never review a partially collected PR set.
    fresh = select_prs(
        selected, session.gh.pulls(), set(children) or {parent.number}, session.gh.repo
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
        ],
        "prs": details,
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
    if result["error"]:
        text += "\n\n未完了: " + result["error"]
    if result["exception"]:
        text += (
            "\n\n通常1回のレビュー予算を超える例外。理由: "
            + result["exception"]["reason"]
        )
    text += "\n\n横断指摘:\n" + finding_text(result["cross_findings"])
    by_pr = {}
    for pr in result["prs"]:
        n = pr["number"]
        snap = next(s for s in result["request"]["prs"] if s["number"] == n)
        own = (
            f"PR #{n}: {pr['coverage']}。修正担当: {snap['branch']} / 子issue {pr['child_issues']}\n\n{pr['summary']}\n\n"
            + finding_text(pr_findings(result, n))
        )
        by_pr[n] = own
        text += "\n\n" + own
    return text, by_pr


def publish(session: ParentSession, run_dir: Path, result: dict[str, Any]) -> str:
    """Publish projections first, then commit the parent record. Pending is not evidence."""
    summary, by_pr = summaries(result)
    body = comment_body(result, summary)
    session.public(body)
    projections = {}
    for pr in result["prs"]:
        n = pr["number"]
        record = {
            "schema_version": 1,
            "kind": "pr",
            "review_run_id": result["review_run_id"],
            "parent_issue": session.parent,
            "pr": n,
            "request_sha256": result["request_sha256"],
            "result_sha256": digest(result),
            "snapshot": next(p for p in result["request"]["prs"] if p["number"] == n),
            "coverage": pr["coverage"],
            "findings": pr_findings(result, n),
        }
        projections[n] = record
        session.public(comment_body(record, by_pr[n]))
    pending = {
        "schema_version": 1,
        "kind": "pending",
        "review_run_id": result["review_run_id"],
        "request_sha256": result["request_sha256"],
    }
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
    comment_id = comment["id"]
    if type(comment_id) is not int or comment_id < 1:
        raise ValueError("GitHub returned an invalid parent comment ID")
    url = f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{comment_id}"
    for n, projection in projections.items():
        pr_body = comment_body(
            {**projection, "parent_run_url": url},
            f"[親runの結果]({url})\n\n" + by_pr[n],
        )
        session.public(pr_body)
        (run_dir / f"pr-{n}-comment.md").write_text(pr_body, encoding="utf-8")
        session.gh.write(
            "POST", f"repos/{session.gh.repo}/issues/{n}/comments", {"body": pr_body}
        )
    session.gh.write(
        "PATCH", f"repos/{session.gh.repo}/issues/comments/{comment_id}", {"body": body}
    )
    (run_dir / "parent-comment.md").write_text(body, encoding="utf-8")
    return url


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    request = request_manifest(read_json(args.request))
    gh = GitHub(request["repo"])
    identity = session_identity(dict(os.environ))
    store = Store(args.state_dir)
    run_dir = store.directory / "reviews" / request["review_run_id"]
    with store.locked():
        session = ParentSession(gh, store, request["parent_issue"], identity)
        auth = authorized(session, request)
        if read_json(run_dir / "request.json") != request:
            raise ValueError("local request snapshot does not match")
        execution_file = run_dir / "execution.json"
        if execution_file.exists():
            execution = read_json(execution_file)
            if execution["request_sha256"] != digest(request):
                raise ValueError("run ID was already used with a different snapshot")
            if execution["status"] == "running":
                raise ValueError(
                    "run is running/interrupted (未完了); never retried automatically; issue a new explicit run ID with reason"
                )
            collect(session, request)
            return {
                "reused": True,
                "result": validate_result(read_json(run_dir / "result.json")),
                "publication": execution["status"],
            }
        authorized(session, request, launching=True)
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
            "status": "running",
            "started_at": now(),
            "authorization": auth,
            "session": identity,
            "agent_invocations": 0,
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
    result = result_record(request, execution["started_at"], previous)
    try:
        context, diffs = collect(session, request)
        Store.write_json(run_dir / "context.json", context)
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        template = (HERE / "prompt_template.md").read_text(encoding="utf-8")
        prompt = build_prompt(request, context, diffs, template)
        prompt_file, output_file = run_dir / "prompt.md", run_dir / "agent_output.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        # Every accepted run reaches this call at most once. No per-PR agent loop.
        with workspace(root, request, context, diffs) as cwd:
            with store.locked():
                if authorized(session, request) != auth:
                    raise ValueError("parent authorization changed before review")
                execution["agent_invocations"] = 1
                Store.write_json(execution_file, execution)
            run_process_group(
                [
                    str(args.agent_exec),
                    "--agent",
                    "codex",
                    "--profile",
                    "review",
                    "--cwd",
                    str(cwd),
                    "--prompt-file",
                    str(prompt_file),
                    "--out",
                    str(output_file),
                    "--mode",
                    "read-only",
                    "--log-dir",
                    str(run_dir),
                    "--run-dir",
                    str(run_dir / "agent-runtime"),
                ],
                cwd=root,
                timeout=args.agent_timeout,
            )
        parsed = parse_output(
            output_file.read_text(encoding="utf-8"),
            [p["number"] for p in request["prs"]],
        )
        result.update(parsed)
        for pr in result["prs"]:
            pr["child_issues"] = next(
                p["child_issues"] for p in context["prs"] if p["number"] == pr["number"]
            )
        if parsed["verdict"] != "incomplete":
            result["status"] = "succeeded"
        else:
            result["error"] = (
                "部分収集・未レビューの対象があります。対象の縮約／親判断待ち。"
            )
        with store.locked():
            if authorized(session, request) != auth:
                raise ValueError("parent authorization changed during review")
            collect(session, request)
    except (Exception, KeyboardInterrupt) as exc:
        (run_dir / "error.log").write_text(str(exc), encoding="utf-8")
        result = result_record(request, execution["started_at"], previous)
        result["error"] = public_error(session, exc)
    result["finished_at"] = now()
    Store.write_json(run_dir / "result.json", result)
    try:
        url = publish(session, run_dir, result)
        execution.update(status="published", parent_run_url=url)
    except (Exception, KeyboardInterrupt) as exc:
        (run_dir / "publication-error.log").write_text(str(exc), encoding="utf-8")
        execution["status"] = "publication_failed"
        result = result_record(request, execution["started_at"], previous)
        result["error"] = (
            "結果の公開に失敗（部分公開を含む）。成功扱いにせず、新しい明示指示を待ちます。 "
            + public_error(session, exc)
        )
        Store.write_json(run_dir / "result.json", result)
        # A pending comment is never accepted by the gate. Best-effort failure report.
        try:
            failure = comment_body(result, summaries(result)[0])
            session.public(failure)
            gh.write("POST", session.endpoint + "/comments", {"body": failure})
        except Exception as notification_error:
            execution["failure_notification_error"] = public_error(
                session, notification_error
            )
    execution["finished_at"] = now()
    Store.write_json(execution_file, execution)
    return {"reused": False, "result": result, "publication": execution["status"]}


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


def resolve(args: argparse.Namespace) -> dict[str, Any]:
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
        if execution["status"] != "published" or result["status"] != "succeeded":
            raise ValueError("cannot resolve a failed/unpublished review")
        originals = {p["number"]: p for p in request["prs"]}
        if args.pr not in originals:
            raise ValueError("PR is outside the reviewed set")
        if args.finding not in {f["id"] for f in pr_findings(result, args.pr)}:
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
        if any(
            original[key] != current[key]
            for key in ("base_ref", "merge_base_sha", "branch")
        ):
            raise ValueError(
                "base ref/merge-base/branch changed; a new explicit review is required"
            )
        advances: tuple[BaseAdvance, ...] = ()
        if original["base_sha"] != current["base_sha"]:
            peers = [
                object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
                for n in originals
                if n != args.pr
            ]
            advances = verify_base_advances(result, [current], peers)
        git("merge-base", "--is-ancestor", original["head_sha"], fixed)
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
    run.add_argument("--request", required=True, type=Path)
    run.add_argument(
        "--agent-exec", type=Path, default=HERE.parent / "lib/agent_exec.sh"
    )
    run.add_argument("--agent-timeout", type=float, default=1800)
    fix = commands.add_parser("resolve")
    fix.add_argument("--run", required=True)
    fix.add_argument("--pr", type=int, required=True)
    fix.add_argument("--finding", required=True)
    fix.add_argument("--fixed-head", required=True)
    fix.add_argument("--evidence", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run" and (
            args.agent_timeout <= 0 or not math.isfinite(args.agent_timeout)
        ):
            raise ValueError("agent-timeout must be positive and finite")
        value = run_review(args) if args.command == "run" else resolve(args)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if args.command == "run":
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
