"""Explicit parent orchestration. Writes only when an agent calls a subcommand."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

from decision_logs import parent_commands as decision_capture
from decision_logs.capture import CURRENT
from lib.agent_settings import implementation_models
from lib.identity import session_identity
from worker_logs.sources import read_original

from . import metrics
from .agreement_comments import record_ignored
from .execution import execution_input, record_autonomous, validate_candidate
from .github import GitHub
from .log_publication import BEFORE_WORKER, Continuation, read_only
from .policy import issue_number, nonempty, number, object_value
from .proposals import receive
from .session import ParentSession
from .state import Store, state_directory, unique_object


def read_text(path: str) -> str:
    return nonempty(Path(path).read_text(encoding="utf-8"), path)


def saved_note(session: ParentSession, reference: str) -> str:
    from .policy import digest

    history = session.gh.record_history(session.parent, session.login)
    history.require_valid()
    notes = [
        r
        for r in history.records
        if r.payload.get("event") == "note"
        and r.publisher_verified
        and reference in (r.url, r.payload.get("body_sha256"))
    ]
    if not notes:
        raise ValueError("evidence-ref must identify a verified note of this parent")
    values = set()
    for note in notes:
        _, separator, body = note.body.partition("\n\n")
        if not separator or digest(body) != note.payload["body_sha256"]:
            raise ValueError("saved evidence content changed")
        values.add(body)
    if len(values) != 1:
        raise ValueError("ambiguous evidence reference")
    return values.pop()


def read_object(path: str) -> dict[str, Any]:
    return object_value(json.loads(read_text(path), object_pairs_hook=unique_object))


WORK_ITEM_HELP = (
    "担当Issue番号。正式な子なしでは --parent と同じ番号を指定（子の新設ではない）"
)


def parser() -> argparse.ArgumentParser:
    from .cli_help import configure

    root = argparse.ArgumentParser(
        prog="parent",
        description="親Issueの担当、委任、正式レビュー、受入を管理。共通引数は操作名の前に指定する。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    try:
        models = implementation_models()
    except (OSError, ValueError) as exc:
        root.error(str(exc))
    root.add_argument("--repo", help="対象のowner/repo。省略時はgit remoteから特定")
    root.add_argument(
        "--state-dir",
        type=Path,
        help="親registryの場所。自律親はbootstrapのparent_state_dirを指定",
    )
    root.add_argument(
        "--dispatcher-state-dir",
        type=Path,
        help="自律親のワーカー操作に使うbootstrapのdispatcher_state_dir",
    )
    root.add_argument(
        "--parent",
        help="担当する親Issue番号またはGitHub Issue URL",
    )
    commands = root.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("register").add_argument(
        "--summary",
        help="sidebar summary (<= 24 display columns); required unless already saved",
    )
    commands.add_parser("summary").add_argument("--text", required=True)
    for command in (
        "status",
        "checking",
        "wait-supplement",
        "reconcile",
        "sync",
        "state-migrate",
    ):
        commands.add_parser(command)
    commands.add_parser("note").add_argument("--body-file", required=True)
    commands.add_parser("notify").add_argument("--spec", required=True)
    commands.add_parser("evidence").add_argument("--create", action="store_true")
    publication = commands.add_parser("publish-decision-log")
    publication.add_argument("--dry-run", action="store_true")
    publication.add_argument("--retry", action="store_true")
    publication.add_argument(
        "--unassigned-parent",
        type=int,
        help="publish another unassigned parent's records as this active parent",
    )
    for name in ("worker-log-status", "publish-worker-log"):
        log = commands.add_parser(name)
        if name == "publish-worker-log":
            targets = log.add_mutually_exclusive_group(required=True)
            targets.add_argument("--child", type=int, help=WORK_ITEM_HELP)
            targets.add_argument("--all-parents", action="store_true")
            targets.add_argument("--adopt-orphan-prs", action="store_true")
            log.add_argument("--source-parent", type=int, action="append")
            log.add_argument("--pr", type=int, nargs="+")
        else:
            log.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
        log.add_argument("--turn", type=int)
        log.add_argument("--delegation-key")
        if name == "publish-worker-log":
            log.add_argument("--dry-run", action="store_true")
            log.add_argument(
                "--retry",
                action="store_true",
                help="recheck a repaired blocked publication without changing its frozen input",
            )
    from project_progress.schema import EVENTS, OWNER, PROJECT_NUMBER

    for command in ("project-setup", "project-sync"):
        project = commands.add_parser(command)
        project.add_argument("--owner", default=OWNER)
        project.add_argument("--project", type=int, default=PROJECT_NUMBER)
        mode = project.add_mutually_exclusive_group()
        mode.add_argument(
            "--apply", action="store_true", help="write the planned Project changes"
        )
        mode.add_argument(
            "--dry-run", action="store_true", help="print the plan (default)"
        )
        if command == "project-sync":
            project.add_argument("--event", choices=EVENTS, required=True)
            project.add_argument("--spec", required=True)
    commands.add_parser("workflow-report").add_argument("--spec", required=True)
    cancellation = commands.add_parser("record-cancellation")
    cancellation.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    cancellation.add_argument("--spec", required=True)
    for name in ("start-worker", "continue-worker", "inspect-worker", "stop-worker"):
        worker = commands.add_parser(name)
        worker.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
        if name in {"start-worker", "continue-worker"}:
            worker.add_argument(
                "--prompt-file",
                required=name == "start-worker",
                help="task/fix instructions; pending planning answers are sent automatically",
            )
            worker.add_argument(
                "--additional-instruction-file",
                help="extra instructions appended separately to a pending planning answer",
            )
            if name == "start-worker":
                worker.add_argument("--model", choices=models)
            else:
                worker.add_argument("--model", help="must match the saved worker model")
            worker.add_argument(
                "--planning",
                action="store_true",
                help="investigate and report before implementation",
            )
        if name == "start-worker":
            worker.add_argument("--branch")
            worker.add_argument("--worktree", type=Path)
            worker.add_argument(
                "--from",
                dest="base_ref",
                help="基準commitから専用branchとworktreeを準備する",
            )
        if name == "continue-worker":
            worker.add_argument("--evidence-file")
        if name == "stop-worker":
            worker.add_argument("--spec", required=True)
    inputs = commands.add_parser("basis-input")
    inputs.add_argument("--ignored-comments")
    inputs.add_argument("--context")
    commands.add_parser("basis-record").add_argument("--spec", required=True)
    candidate = commands.add_parser("basis-check")
    candidate.add_argument("--basis-sha256", required=True)
    candidate.add_argument("--generation", type=int, required=True)
    commands.add_parser("basis-revoke").add_argument("--reason-file", required=True)
    commands.add_parser("agreement-ignore-comments").add_argument(
        "--ignored-comments", required=True
    )
    release = commands.add_parser("release")
    release.add_argument("--ended", action="store_true")
    release_input = release.add_mutually_exclusive_group(required=True)
    release_input.add_argument("--note-file")
    release_input.add_argument(
        "--spec", help="structured acceptance and next improvements; requires --ended"
    )
    handoff = commands.add_parser("handoff")
    handoff.add_argument("--accept", action="store_true")
    handoff.add_argument(
        "--summary",
        help="sidebar summary for --accept; required unless already saved",
    )
    handoff.add_argument(
        "--recover",
        action="store_true",
        help="explicit takeover after inspecting an unexpectedly interrupted owner",
    )
    handoff.add_argument("--note-file", required=True)
    agree = commands.add_parser("agree")
    agree_input = agree.add_mutually_exclusive_group(required=True)
    agree_input.add_argument("--spec")
    agree_input.add_argument("--improvement-approval")
    agree.add_argument("--report", type=int)
    agree.add_argument("--proposal")
    commands.add_parser("create-child").add_argument("--spec", required=True)
    receipt = commands.add_parser("receive-proposal")
    receipt.add_argument("--report", type=int, required=True)
    receipt.add_argument("--proposal", required=True)
    delegate = commands.add_parser("delegate")
    delegate.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    delegate.add_argument("--agent", choices=("codex",), required=True)
    delegate.add_argument("--branch", required=True)
    delegate.add_argument(
        "--worktree",
        type=Path,
        required=True,
        help="メインcheckoutの .claude/worktrees 配下に作成済みの専用worktree",
    )
    delegate.add_argument("--dependencies", type=int, nargs="*")
    delegate.add_argument(
        "--restart-evidence-file",
        help="再予約または旧Astraからの引継ぎで、元の担当の終了と引継ぎ内容を記したファイル",
    )
    result = commands.add_parser("record-result")
    result.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    result.add_argument("--evidence-file", required=True)
    policy = commands.add_parser("configure-delegation")
    policy.add_argument(
        "--max-parallel-children", required=True, help="all or a positive integer"
    )
    policy.add_argument("--reason-file", required=True)
    ended = commands.add_parser("record-worker-ended")
    ended.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    ended.add_argument("--delegation-id", required=True)
    ended.add_argument("--evidence-file", required=True)
    manual = commands.add_parser("run-worker")
    manual.add_argument(
        "--from",
        dest="base_ref",
        help="新規担当のworktree準備・委任・初回起動をまとめる",
    )
    manual.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    manual.add_argument(
        "--prompt-file",
        type=Path,
        help="task/fix instructions; pending planning answers are sent automatically",
    )
    manual.add_argument("--additional-instruction-file")
    continuation = manual.add_mutually_exclusive_group()
    continuation.add_argument("--resume", action="store_true")
    continuation.add_argument(
        "--replace-session",
        action="store_true",
        help="start a new session with explicit evidence that the saved session cannot resume",
    )
    manual.add_argument(
        "--planning",
        action="store_true",
        help="investigate and report before implementation",
    )
    manual.add_argument("--restart-evidence-file")
    manual.add_argument("--repair-id")
    manual.add_argument("--repair-pr", type=int)
    manual.add_argument(
        "--model",
        help=f"initial model ({', '.join(models)}); resume inherits the saved model",
    )
    manual.add_argument(
        "--model-reason-file",
        help="optional note retained with the worker record",
    )
    run_record = commands.add_parser("record-worker-run")
    run_record.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    run_record.add_argument("--run-dir", type=Path, required=True)
    wait = commands.add_parser(
        "wait-worker", help="指定ワーカーの終了・回収と手動委任の後続処理を待つ"
    )
    wait.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    wait.add_argument(
        "--timeout", type=float, help="finite positive seconds; default: no limit"
    )
    decision = commands.add_parser(
        "decide-worker-plan",
        help="bind a parent answer to the latest completed planning final",
    )
    decision.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    decision.add_argument(
        "--action", choices=("approve", "investigate", "reject"), required=True
    )
    decision.add_argument("--answer-file", required=True)
    refresh = commands.add_parser(
        "refresh-worker-plan",
        help="refresh the saved controller; normal continuation does this automatically",
    )
    refresh.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    refresh.add_argument("--evidence-file", required=True)
    for command in ("stop", "resume"):
        sub = commands.add_parser(command)
        sub.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
        sub.add_argument("--spec", required=True)
    improvement = commands.add_parser("improvement")
    improvement.add_argument("--child", type=int)
    improvement.add_argument("--key")
    improvement.add_argument("--spec", help="候補の詳細JSON。- は標準入力")
    improvement.add_argument("--purpose", help="改善の目的")
    improvement.add_argument("--evidence", help="観測した問題の根拠")
    improvement.add_argument("--scope", help="改善する範囲")
    commands.add_parser("improvement-propose")
    extract = commands.add_parser("improvement-extract")
    extract.add_argument("--candidates", nargs="+", required=True)
    expire = commands.add_parser("improvement-expire")
    expire.add_argument("--now")
    expire.add_argument("--dry-run", action="store_true")
    decide = commands.add_parser("improvement-decide")
    decide.add_argument("--candidate", action="append", required=True)
    decide.add_argument(
        "--action", choices=("approve", "hold", "decline", "revoke"), required=True
    )
    decide.add_argument("--spec", required=True)
    from .review_choice import add_arguments as review_choice_arguments

    inspection = commands.add_parser("review-inspect")
    inspection.add_argument("--prs", type=int, nargs="+", required=True)
    request = commands.add_parser("review-request")
    request.add_argument("--prs", type=int, nargs="+", required=True)
    evidence = request.add_mutually_exclusive_group(required=True)
    evidence.add_argument(
        "--evidence-ref", help="この親のnoteが返したbody_sha256またはURL"
    )
    evidence.add_argument(
        "--evidence-file",
        help="受入条件・対象HEAD・テスト結果を記したファイル",
    )
    request.add_argument("--reason-file")
    instruction = request.add_mutually_exclusive_group(required=True)
    instruction.add_argument("--instruction", help="レビュー指示を直接指定")
    instruction.add_argument(
        "--instruction-file",
        help="原要件とレビューで確認すべき点を記したファイル",
    )
    request.add_argument(
        "--wait", action="store_true", help="最初の通知または完了まで待つ"
    )
    request.add_argument("--agent-exec", type=Path)
    request.add_argument("--agent-timeout", type=float, default=1800)
    request.add_argument("--cpu-tests-file", type=Path)
    request.add_argument("--isolation-proof", type=Path)
    review_choice_arguments(request)
    status = commands.add_parser("merge-status")
    status.add_argument("--run-id")
    confirmation = commands.add_parser("merge-confirm")
    confirmation.add_argument("--prs", nargs="+", type=int, required=True)
    confirmation.add_argument("--wait", action="store_true")
    resume = commands.add_parser("merge-continue")
    resume.add_argument("--run-id")
    resume.add_argument(
        "--instruction-file",
        required=True,
        help="majorごとの対処・検証または続行理由を記したファイル",
    )
    resume.add_argument("--prs", nargs="+", type=int)
    resume.add_argument("--wait", action="store_true")
    waiting = commands.add_parser("wait-merge")
    waiting.add_argument("--run-id")
    waiting.add_argument("--after", type=int, default=0)
    waiting.add_argument("--timeout", type=float)
    for name in (
        "delegate",
        "run-worker",
        "wait-worker",
        "record-worker-ended",
        "record-worker-run",
        "start-worker",
        "continue-worker",
        "inspect-worker",
        "stop-worker",
        "decide-worker-plan",
        "refresh-worker-plan",
        "worker-log-status",
        "publish-worker-log",
    ):
        commands.choices[name].add_argument(
            "--worker",
            help="統合担当はprimary、補助担当は委任時に付けた名前。複数担当の操作時に指定する",
        )
    for name in ("delegate", "start-worker"):
        commands.choices[name].add_argument(
            "--handoff-from-astra",
            metavar="DELEGATION_ID",
            help="終了した旧Astra担当の委任ID。同じPR・branch・worktreeを新しいSol担当へ移す",
        )
    commands.choices["start-worker"].add_argument(
        "--handoff-evidence-file",
        help="旧Astraからの引継ぎ理由・成果と未処理指摘の記録",
    )
    decision_capture.add_parsers(commands)
    configure(root, commands.choices)
    return root


def _main(argv: argparse.Namespace | list[str] | None = None) -> int:
    args = argv if isinstance(argv, argparse.Namespace) else parser().parse_args(argv)
    capture_token = None
    continuation = None
    try:
        if args.command == "publish-worker-log":
            if (args.source_parent and not args.all_parents) or (
                args.pr and not args.adopt_orphan_prs
            ):
                raise ValueError(
                    "--source-parent requires --all-parents; --pr requires --adopt-orphan-prs"
                )
            if (args.all_parents or args.adopt_orphan_prs) and (
                args.turn is not None
                or args.delegation_key is not None
                or args.worker is not None
                or args.retry
            ):
                raise ValueError(
                    "bulk recovery does not accept --turn, --delegation-key or --retry"
                )
        if args.command == "state-migrate":
            with Store(args.state_dir or state_directory()).locked() as migration_store:
                migration_store.migrate()
                print(
                    json.dumps(
                        {
                            "version": migration_store.data["version"],
                            "parents": len(migration_store.parents),
                        }
                    )
                )
                return 0
        if args.command == "handoff" and args.recover and not args.accept:
            raise ValueError("--recover requires handoff --accept")
        # Missing identity is a configuration error before any network/state write.
        identity = (
            None
            if (
                args.command in {"publish-worker-log", "publish-decision-log"}
                and getattr(args, "dry_run", False)
                and getattr(args, "unassigned_parent", None) is None
            )
            or args.command
            in {
                "status",
                "basis-input",
                "basis-record",
                "basis-check",
                "wait-supplement",
                "wait-worker",
                "wait-merge",
                "merge-status",
                "worker-log-status",
                "review-inspect",
                "decision-log-status",
                "evidence",
                "improvement-expire",
            }
            else session_identity(dict(os.environ))
        )
        args.state_dir = (args.state_dir or state_directory()).resolve()
        workers = {"start-worker", "continue-worker", "inspect-worker", "stop-worker"}
        store: Store
        if args.command in workers | {
            "workflow-report",
            "record-cancellation",
            "improvement-decide",
        }:
            from .development import workflow_store

            store = workflow_store(args.state_dir, args.dispatcher_state_dir)
        else:
            store = Store(args.state_dir)
        gh = GitHub(
            args.repo,
            store=None if read_only(args) else store,
        )
        # Inspection and waiting have always used a fresh, unbound reader.
        # Other read-only commands keep their former per-transaction reuse.
        gh._read_only_cache = read_only(args) and args.command not in {
            "review-inspect",
            "wait-supplement",
            "wait-worker",
        }
        if args.command == "improvement-expire":
            from .improvement_proposals import expire
            from .improvements import lock_key

            with (
                store.operation(lock_key(gh.repo)),
                store.consistent_snapshot(),
            ):
                # Expiry only reads the registry for public-payload validation.
                # Its business state is canonical comments and creation receipts.
                # The snapshot read excludes peer saves; GitHub I/O runs after it.
                outcome = expire(
                    gh.detached(), store, now=args.now, dry_run=args.dry_run
                )
            print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))
            return 2 if outcome["errors"] else 0
        if args.parent is None:
            raise ValueError("--parent is required for this command")
        parent = issue_number(args.parent, gh.repo)
        if args.command in {"merge-status", "wait-merge"}:
            from lib.ops_review.merge_gate_impl import latest, view, wait

            path = latest(store, gh.repo, parent, args.run_id)
            value = (
                wait(path, after=args.after, timeout=args.timeout)
                if args.command == "wait-merge"
                else view(path)
            )
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "wait-worker":
            from .worker_wait import wait_worker

            code, value = wait_worker(
                store,
                f"{gh.repo.casefold()}#{parent}",
                number(args.child),
                timeout=args.timeout,
                worker=args.worker,
            )
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return code
        continuation = Continuation(args, store, gh.repo, parent, identity)
        if args.command == "publish-worker-log" and (
            args.all_parents or args.adopt_orphan_prs
        ):
            from worker_logs.reconciliation import bulk_command

            value = bulk_command(
                gh.repo, parent, identity, store=store, args=args, root=Path.cwd()
            )
            code = (
                2
                if value.get("error") or any(r.get("error") for r in value["logs"])
                else 0
            )
            continuation.recovery_visits(value)
            value = continuation.complete(args, value, exit_code=code)
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
            return code
        if args.command == "publish-decision-log":
            from decision_logs.publication import Publisher
            from worker_logs.reconciliation import decision_journal

            # Publication bookkeeping never enters the capture adapter: an
            # otherwise successful publication must not create another batch.
            if args.unassigned_parent is not None:
                from .log_publication import publish_unassigned_decisions

                value = publish_unassigned_decisions(
                    args, continuation, root=Path.cwd()
                )
                code = 2 if value.get("error") or value.get("blocked") else 0
                print(json.dumps(value, ensure_ascii=False, sort_keys=True))
                return code
            owner = (
                store.read_data()
                .get("parents", {})
                .get(f"{gh.repo.casefold()}#{parent}", {})
            )
            if owner.get("session") != identity and not args.dry_run:
                raise ValueError("decision publication requires the assigned parent")
            value = Publisher(
                Path.cwd(), decision_journal(gh.repo, parent, state_dir=store.directory)
            ).publish(dry_run=args.dry_run, retry=args.retry)
            code = 2 if value["error"] or value["blocked"] else 0
            value = continuation.complete(args, value, exit_code=code)
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
            return code
        if args.command == "review-inspect":
            import importlib

            inspection_module = importlib.import_module(
                "lib.ops_review.review_inspection"
            )
            inspection = inspection_module.inspect_reviews(gh, store, parent, args.prs)
            print(json.dumps(inspection, ensure_ascii=False, indent=2, sort_keys=True))
            return (
                2
                if inspection["errors"]
                or any(
                    pr["errors"] or any(run["errors"] for run in pr["history"])
                    for pr in inspection["prs"]
                )
                else 0
            )
        if args.command in {"decision-log-status", "retry-decision-capture"}:
            from decision_logs.capture import status as capture_status
            from worker_logs.reconciliation import decision_resume_command

            if args.command == "retry-decision-capture":
                row = (
                    store.read_data()
                    .get("parents", {})
                    .get(f"{gh.repo.casefold()}#{parent}", {})
                )
                if row.get("session") != identity:
                    raise ValueError(
                        "decision capture recovery requires the assigned parent"
                    )
            value = capture_status(
                store.directory,
                gh.repo,
                parent,
                retry=args.command == "retry-decision-capture",
            )
            with store.snapshot():
                row = store.parents.get(f"{gh.repo.casefold()}#{parent}", {})
                value["decision_resume_command"] = decision_resume_command(
                    gh.repo,
                    parent,
                    assigned=row.get("status") in {"active", "handoff_pending"},
                )
            continuation.sweep()
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
            return 0 if value["status"] == "captured" else 2
        if args.command in BEFORE_WORKER:
            continuation.sweep()
        capture_source = not read_only(args)
        if args.command in {
            "improvement",
            "improvement-propose",
            "improvement-extract",
            "improvement-decide",
        }:
            owner = (
                store.read_data()
                .get("parents", {})
                .get(f"{gh.repo.casefold()}#{parent}", {})
            )
            capture_source = capture_source and (
                owner.get("status") == "active" and owner.get("session") == identity
            )
        capture_token = CURRENT.set(
            decision_capture.begin(args, store, gh.repo, parent)
            if capture_source
            else None
        )
        if args.command == "merge-continue":
            from lib.ops_review.merge_gate_impl import (
                continue_progression,
                latest,
                view,
                wait,
            )

            path = latest(store, gh.repo, parent, args.run_id)
            after = len(view(path)["events"])
            continue_progression(
                path,
                store,
                nonempty(identity, "parent identity"),
                read_text(args.instruction_file),
                prs=args.prs,
            )
            value = wait(path, after=after) if args.wait else view(path)
            value = continuation.complete(args, value)
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command in {"worker-log-status", "publish-worker-log"}:
            from worker_logs.service import command as log_command

            log_result = log_command(
                gh,
                store,
                parent,
                number(args.child),
                identity,
                publish_logs=args.command == "publish-worker-log",
                sequence=number(args.turn) if args.turn is not None else None,
                key=args.delegation_key,
                worker=args.worker,
                dry_run=getattr(args, "dry_run", False),
                retry=getattr(args, "retry", False),
            )
            code = (
                2
                if log_result["unavailable_logs"]
                or any(r.get("error") for r in log_result["logs"])
                else 0
            )
            log_result = continuation.complete(args, log_result, exit_code=code)
            print(json.dumps(log_result, ensure_ascii=False, sort_keys=True))
            return code
        if args.command == "wait-supplement":
            from .supplement import wait_supplement

            # Waiting does not need ownership, a registry transaction or an operation lock.
            print(
                json.dumps(
                    wait_supplement(gh, parent), ensure_ascii=False, sort_keys=True
                )
            )
            return 0
        if args.command == "run-worker":
            from lib.repair_turn import request as repair_request

            from .manual_workers import run_worker

            code, worker_result = run_worker(
                gh,
                store,
                parent,
                nonempty(identity, "parent identity"),
                child=number(args.child),
                prompt=args.prompt_file,
                base_ref=args.base_ref,
                resume=args.resume,
                worker=args.worker,
                repair=repair_request(args.repair_id, args.repair_pr)
                if args.repair_id is not None or args.repair_pr is not None
                else None,
                replace_session=args.replace_session,
                planning=args.planning,
                model=args.model,
                model_reason=read_text(args.model_reason_file)
                if args.model_reason_file
                else None,
                additional=read_original(args.additional_instruction_file)
                if args.additional_instruction_file
                else None,
                restart_evidence=read_text(args.restart_evidence_file)
                if args.restart_evidence_file
                else None,
            )
            worker_result = continuation.complete(args, worker_result, exit_code=code)
            print(
                json.dumps(worker_result, ensure_ascii=False, indent=2, sort_keys=True)
            )
            return code
        if args.command in workers:
            from .development import worker_command

            with (
                nullcontext()
                if args.command != "stop-worker"
                else store.operation(f"{gh.repo.casefold()}#{parent}")
            ):
                worker_result = worker_command(
                    args, gh, parent, nonempty(identity, "parent identity"), store
                )
            worker_result = continuation.complete(args, worker_result)
            print(
                json.dumps(worker_result, ensure_ascii=False, indent=2, sort_keys=True)
            )
            return 0
        operation = (
            nullcontext()
            if read_only(args) or args.command == "reconcile"
            else store.operation(f"{gh.repo.casefold()}#{parent}")
        )
        project_operation = (
            store.operation(f"project:{args.owner.casefold()}#{args.project}")
            if args.command in {"project-setup", "project-sync"} and not read_only(args)
            else nullcontext()
        )
        ownership = (
            store.ownership()
            if args.command in {"register", "handoff", "release"}
            else nullcontext()
        )
        transaction = store.snapshot() if read_only(args) else store.locked()
        history_reads = gh.history_read_scope()
        from .improvements import lock_key

        improvement_operation = (
            store.operation(lock_key(gh.repo))
            if args.command
            in {
                "improvement",
                "improvement-propose",
                "improvement-extract",
                "improvement-decide",
            }
            else nullcontext()
        )
        with (
            improvement_operation,
            operation,
            project_operation,
            ownership,
            transaction,
            history_reads,
        ):
            session = ParentSession(gh, store, parent, identity)
            if hasattr(args, "worker"):
                from .worker_slots import select

                session.worker = select(
                    session.row,
                    args.child,
                    args.worker,
                    creating=args.command == "delegate",
                )
            command = args.command
            result: Any
            if command in {"project-setup", "project-sync"}:
                from project_progress.operations import parent_command

                result = parent_command(
                    session,
                    command,
                    args.owner,
                    args.project,
                    args.apply,
                    args.event if command == "project-sync" else None,
                    read_object(args.spec) if command == "project-sync" else None,
                )
            elif command == "decide":
                result = decision_capture.decide(session, args.spec)
            elif command == "basis-input":
                ignored = (
                    json.loads(read_text(args.ignored_comments))
                    if args.ignored_comments
                    else []
                )
                if not isinstance(ignored, list):
                    raise ValueError("ignored comments must be a list")
                result = execution_input(
                    session,
                    ignored,
                    read_object(args.context) if args.context else None,
                )
            elif command == "basis-record":
                if any(
                    os.environ.get(name, "").strip()
                    for name in ("CLAUDE_CODE_SESSION_ID", "CODEX_SESSION_ID")
                ):
                    session.session = session_identity(dict(os.environ))
                result = record_autonomous(session, read_object(args.spec))
            elif command == "agreement-ignore-comments":
                result = record_ignored(
                    session, json.loads(read_text(args.ignored_comments))
                )
            elif command == "basis-revoke":
                result = session.revoke_basis(read_text(args.reason_file))
            elif command == "basis-check":
                result = validate_candidate(session, args.basis_sha256, args.generation)
            elif command == "register":
                result = session.register(args.summary)
            elif command == "summary":
                result = session.set_summary(args.text)
            elif command == "status":
                # Status works outside a live session; disclose no raw session identifiers.
                with suppress(ValueError):
                    session.session = session_identity(dict(os.environ))
                result = session.status()
            elif command == "checking":
                result = session.checking()
            elif command == "evidence":
                result = gh.evidence_issues().describe(
                    parent, session.login, create=args.create
                )
            elif command == "note":
                result = session.note(read_text(args.body_file))
                session.sync_progress()
            elif command == "notify":
                from .notifications import notify

                result = notify(session, read_object(args.spec))
            elif command == "workflow-report":
                from .development import load, report

                spec = load(Path(args.spec))
                if spec.get("schema_version") == 2 and "acceptance" in spec:
                    from .completion import submit

                    result = submit(session, spec, managed_report=True)
                else:
                    result = report(session, spec)
            elif command == "record-cancellation":
                from .development import load, record_cancellation

                result = record_cancellation(
                    session, number(args.child), load(Path(args.spec))
                )
            elif command == "agree":
                result = session.agree(
                    read_object(args.spec) if args.spec else None,
                    args.report,
                    args.proposal,
                    args.improvement_approval,
                )
            elif command == "release":
                if args.spec:
                    if not args.ended:
                        raise ValueError("structured finish requires release --ended")
                    from .completion import release

                    spec = read_object(args.spec)
                    if spec.get("schema_version") == 2:
                        from .completion import submit

                        result = submit(session, spec, managed_report=False)
                    else:
                        result = release(session, spec)
                else:
                    result = session.lifecycle(
                        "ended" if args.ended else "released", read_text(args.note_file)
                    )
            elif command == "handoff":
                if args.summary is not None and not args.accept:
                    raise ValueError("handoff --summary requires --accept")
                note = read_text(args.note_file)
                result = (
                    session.accept(note, args.recover, summary=args.summary)
                    if args.accept
                    else session.lifecycle("handoff_pending", note)
                )
            elif command == "create-child":
                result = session.create_child(read_object(args.spec))
            elif command == "receive-proposal":
                result = receive(session, args.report, args.proposal)
            elif command == "reconcile":
                result = session.reconcile()
            elif command == "sync":
                session.owner()
                result = {
                    "progress": session.sync_progress(),
                    "launch_plan": session.launch_plan(),
                }
            elif command == "configure-delegation":
                if args.max_parallel_children != "all" and not re.fullmatch(
                    r"[1-9][0-9]*", args.max_parallel_children
                ):
                    raise ValueError(
                        "--max-parallel-children must be all or a positive integer"
                    )
                result = session.configure_delegation(
                    None
                    if args.max_parallel_children == "all"
                    else int(args.max_parallel_children),
                    read_text(args.reason_file),
                )
            elif command == "record-worker-ended":
                result = session.record_worker_ended(
                    number(args.child),
                    args.delegation_id,
                    read_text(args.evidence_file),
                )
            elif command == "record-worker-run":
                from .manual_workers import record_worker_run

                result = record_worker_run(session, number(args.child), args.run_dir)
            elif command == "delegate":
                result = session.delegate(
                    number(args.child),
                    args.agent,
                    args.branch,
                    args.worktree,
                    args.dependencies,
                    handoff_from_astra=args.handoff_from_astra,
                    restart=read_text(args.restart_evidence_file)
                    if args.restart_evidence_file
                    else None,
                )
            elif command in {"stop", "resume"}:
                result = session.stop_or_resume(
                    number(args.child), read_object(args.spec), command == "resume"
                )
            elif command == "improvement":
                if args.spec and any((args.purpose, args.evidence, args.scope)):
                    raise ValueError(
                        "choose --spec or --purpose / --evidence / --scope"
                    )
                candidate = (
                    read_object(args.spec)
                    if args.spec
                    else {
                        k: nonempty(getattr(args, k), "--" + k)
                        for k in ("purpose", "evidence", "scope")
                    }
                )
                result = session.improvement(
                    number(args.child) if args.child is not None else None,
                    candidate,
                    args.key,
                )
            elif command == "improvement-propose":
                from .improvement_proposals import propose

                result = propose(session)
            elif command == "improvement-extract":
                from .improvement_proposals import extract

                result = extract(session, args.candidates)
            elif command == "improvement-decide":
                from .improvements import decide

                result = decide(
                    session, args.candidate, args.action, read_object(args.spec)
                )
            elif command in {
                "decide-worker-plan",
                "refresh-worker-plan",
            }:
                from . import worker_preflight

                if command == "decide-worker-plan":
                    result = worker_preflight.decide(
                        session,
                        number(args.child),
                        args.action,
                        read_original(args.answer_file),
                    )
                else:
                    from .development import refresh_worker_plan

                    result = refresh_worker_plan(
                        session,
                        args.dispatcher_state_dir,
                        number(args.child),
                        read_text(args.evidence_file),
                    )
            elif command == "record-result":
                result = session.record_result(
                    number(args.child), read_text(args.evidence_file)
                )
            elif command == "review-request":
                from .review_choice import selection as review_choice
                from .review_handoff import request as handoff_review

                options = {"agent_timeout": args.agent_timeout}
                for name in (
                    "agent_exec",
                    "review_config_file",
                    "isolation_proof",
                    "dispatcher_state_dir",
                ):
                    if getattr(args, name) is not None:
                        options[name] = str(getattr(args, name).resolve())
                if args.cpu_tests_file:
                    options["cpu_tests"] = read_object(str(args.cpu_tests_file))
                result = handoff_review(
                    session,
                    args.prs,
                    args.instruction
                    if args.instruction is not None
                    else read_text(args.instruction_file),
                    saved_note(session, args.evidence_ref)
                    if args.evidence_ref is not None
                    else read_text(args.evidence_file),
                    read_text(args.reason_file) if args.reason_file else None,
                    review_selection=review_choice(session, args),
                    options=options,
                )
            elif command == "merge-confirm":
                from lib.ops_review.post_merge import register as confirm_merged

                result = confirm_merged(session, args.prs)
            else:
                raise ValueError("unknown parent command")
            if command in {
                "checking",
                "agree",
                "basis-revoke",
            }:
                session.sync_progress()
        if command in {"review-request", "merge-confirm"}:
            from lib.ops_review.merge_gate_impl import start, state_path, wait

            path = state_path(store, result["review_run_id"])
            if not result["reused"]:
                start(path)
            if args.wait:
                result = wait(path)
        integration_failed = False
        if command == "improvement-decide":
            from .improvement_runtime import reconcile_stop

            # The decision is durable before stopping; do not hold the source
            # operation or registry lock while a runner terminates.
            stops = [
                reconcile_stop(
                    GitHub(gh.repo, store=store),
                    store,
                    parent,
                    nonempty(identity, "recorder identity"),
                    candidate,
                    args.dispatcher_state_dir,
                )
                for candidate in args.candidate
            ]
            result["stop"] = (
                stops[0]
                if len(stops) == 1
                else {
                    "state": "stopping"
                    if any(s["state"] == "stopping" for s in stops)
                    else "stopped"
                    if any(s["state"] == "stopped" for s in stops)
                    else "not_requested",
                    "candidates": dict(zip(args.candidate, stops, strict=True)),
                }
            )
            integration_failed = result["stop"]["state"] == "stopping"
        if command == "release" and args.ended:
            from .improvements import candidates

            with store.locked():
                source_session = ParentSession(gh, store, parent, identity)
                result["unpublished_improvements"] = [
                    f"I-{parent}-{k}"
                    for k, r in candidates(source_session).items()
                    if r["proposal"] is None and r["publication"] is None
                ]
        if command in {
            "register",
            "summary",
            "reconcile",
            "sync",
            "release",
            "handoff",
        }:
            # The mutation is durable and its registry lock is released. Sync
            # the caller and unnamed active peers before projecting the name.
            result = {**result, "parent_action_applied": True}
            from board.errors import BoardError
            from board.service import sync_parent

            try:
                name_sync = sync_parent(dict(os.environ), gh.repo, parent)
            except BoardError as exc:
                name_sync = {"status": "error", "error": str(exc)}
                integration_failed = True
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ):
                name_sync = {
                    "status": "error",
                    "error": "board name synchronization failed; inspect storage and parent configuration",
                }
                integration_failed = True
            if name_sync["status"] == "error":
                integration_failed = True
            result = {**result, "board_sync": name_sync}
            from herdr_labels.service import sync as sync_display

            try:
                display = sync_display(
                    dict(os.environ),
                    confirm_parent=command == "register"
                    or (command == "handoff" and args.accept),
                    clear=command == "release"
                    or (command == "handoff" and not args.accept),
                )
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ) as exc:
                display = {
                    "status": "error",
                    "error": str(exc),
                    "repair": (
                        "担当操作は適用済みです。表示同期の失敗原因を解消してください。"
                        "表示先が未確認の場合は .agents/ops/herdr_labels/README.md の手順で "
                        "自分のpaneを確認して herdr_labels bind を行ってから、"
                        "PYTHONPATH=.agents/ops .venv/bin/python -m herdr_labels sync "
                        "で表示だけを再試行してください。"
                    ),
                }
                integration_failed = True
            result = {**result, "display_sync": display}
            if name_sync.get("name") and (
                command == "register" or (command == "handoff" and args.accept)
            ):
                from board.lifecycle import restored

                try:
                    delivery = restored(dict(os.environ), name_sync["name"])
                except (
                    BoardError,
                    ValueError,
                    OSError,
                    KeyError,
                    TypeError,
                    subprocess.SubprocessError,
                ):
                    delivery = {
                        "status": "error",
                        "reason": "route_unavailable",
                    }
                if delivery["status"] == "error":
                    integration_failed = True
                result = {**result, "board_delivery": delivery}
        result = continuation.complete(
            args, result, exit_code=1 if integration_failed else 0
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if integration_failed else 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        decision_capture.failed(exc)
        if continuation is not None:
            observed_exit = continuation.failed()
            if observed_exit is not None:
                print(json.dumps(observed_exit, ensure_ascii=False, sort_keys=True))
        # Do not echo arbitrary gh payloads/registry contents containing local IDs.
        print(
            f"parent-session: {exc}\n操作の説明: parent {args.command} --help",
            file=sys.stderr,
        )
        return 2
    finally:
        if capture_token is not None:
            CURRENT.reset(capture_token)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    destination = os.environ.get("PARENT_METRICS_FILE")
    with metrics.command(
        args.command, Path(destination) if destination else None
    ) as row:
        # Several consumers (including the audit capture) may read the same
        # input. Spool stdin once, preserving original bytes and path types.
        streams = [
            key
            for key, value in vars(args).items()
            if (key == "spec" or key.endswith("_file")) and str(value) == "-"
        ]
        if len(streams) > 1:
            print("parent: only one input may consume stdin (-)", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory() if streams else nullcontext() as directory:
            if streams:
                key = streams[0]
                path = Path(str(directory)) / "stdin"
                path.write_bytes(sys.stdin.buffer.read())
                setattr(
                    args,
                    key,
                    path if isinstance(getattr(args, key), Path) else str(path),
                )
            result = _main(args)
        row["exit_code"] = result
        return result
