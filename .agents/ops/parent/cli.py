"""Explicit parent orchestration. Writes only when an agent calls a subcommand."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
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
from .policy import issue_number, nonempty, number, object_value
from .proposals import receive
from .session import ParentSession
from .snapshot import prepare, transition
from .state import Store, state_directory, unique_object


def read_text(path: str) -> str:
    return nonempty(Path(path).read_text(encoding="utf-8"), path)


def read_object(path: str) -> dict[str, Any]:
    return object_value(json.loads(read_text(path), object_pairs_hook=unique_object))


WORK_ITEM_HELP = (
    "担当Issue番号。正式な子なしでは --parent と同じ番号を指定（子の新設ではない）"
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    try:
        models = implementation_models()
    except (OSError, ValueError) as exc:
        root.error(str(exc))
    root.add_argument("--repo", help="owner/repo; default: gh repo view")
    root.add_argument(
        "--state-dir", type=Path, help="the same registry as dispatcher/review/merge"
    )
    root.add_argument("--dispatcher-state-dir", type=Path)
    root.add_argument(
        "--parent",
        required=True,
        help="issue number or https://github.com/owner/repo/issues/N",
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("register").add_argument(
        "--summary", help="sidebar summary (<= 24 display columns)"
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
    publication = commands.add_parser("publish-decision-log")
    publication.add_argument("--dry-run", action="store_true")
    publication.add_argument("--retry", action="store_true")
    for name in ("worker-log-status", "publish-worker-log"):
        log = commands.add_parser(name)
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
            worker.add_argument("--model", choices=models)
            worker.add_argument(
                "--planning",
                action="store_true",
                help="investigate and report before implementation",
            )
        if name == "start-worker":
            worker.add_argument("--branch", required=True)
            worker.add_argument("--worktree", type=Path, required=True)
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
    delegate.add_argument("--agent", choices=("claude", "codex"), required=True)
    delegate.add_argument("--branch", required=True)
    delegate.add_argument("--worktree", type=Path, required=True)
    delegate.add_argument("--dependencies", type=int, nargs="*")
    delegate.add_argument("--restart-evidence-file")
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
    manual.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    manual.add_argument(
        "--prompt-file",
        type=Path,
        help="task/fix instructions; pending planning answers are sent automatically",
    )
    manual.add_argument("--additional-instruction-file")
    manual.add_argument("--resume", action="store_true")
    manual.add_argument(
        "--planning",
        action="store_true",
        help="investigate and report before implementation",
    )
    manual.add_argument("--restart-evidence-file")
    manual.add_argument(
        "--model",
        choices=models,
        help="initial model; resume inherits the saved model and rejects changes",
    )
    manual.add_argument(
        "--model-reason-file",
        help="parent's selection reason; required for an initial alternate model",
    )
    run_record = commands.add_parser("record-worker-run")
    run_record.add_argument("--child", type=int, required=True, help=WORK_ITEM_HELP)
    run_record.add_argument("--run-dir", type=Path, required=True)
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
    improvement.add_argument("--spec", required=True)
    publish = commands.add_parser("improvement-publish")
    publish.add_argument("--candidate", required=True)
    publish.add_argument("--existing", type=int)
    decide = commands.add_parser("improvement-decide")
    decide.add_argument("--candidate", required=True)
    decide.add_argument(
        "--action", choices=("approve", "hold", "decline", "revoke"), required=True
    )
    decide.add_argument("--spec", required=True)
    review = commands.add_parser("review-prepare")
    review.add_argument("--prs", type=int, nargs="+", required=True)
    review.add_argument("--reason-file")
    from .review_choice import add_arguments as review_choice_arguments

    review_choice_arguments(review)
    inspection = commands.add_parser("review-inspect")
    inspection.add_argument("--prs", type=int, nargs="+", required=True)
    request = commands.add_parser("review-request")
    selection = request.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--prs", type=int, nargs="+")
    request.add_argument("--evidence-file")
    request.add_argument("--reason-file")
    request.add_argument("--instruction-file", required=True)
    review_choice_arguments(request)
    state = commands.add_parser("review-state")
    state.add_argument("--run-id", required=True)
    state.add_argument(
        "--state",
        choices=("result_received", "fixes_verified", "complete"),
        required=True,
    )
    state.add_argument("--evidence-file", required=True)
    decision_capture.add_parsers(commands)
    return root


def _main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    capture_token = None
    try:
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
            if args.command
            in {
                "status",
                "basis-input",
                "basis-record",
                "basis-check",
                "wait-supplement",
                "worker-log-status",
                "review-inspect",
                "decision-log-status",
            }
            else session_identity(dict(os.environ))
        )
        args.state_dir = (args.state_dir or state_directory()).resolve()
        workers = {"start-worker", "continue-worker", "inspect-worker", "stop-worker"}
        store: Store
        if args.command in workers | {"workflow-report", "record-cancellation"}:
            from .development import workflow_store

            store = workflow_store(args.state_dir, args.dispatcher_state_dir)
        else:
            store = Store(args.state_dir)
        gh = GitHub(
            args.repo,
            store=None
            if args.command in {"wait-supplement", "review-inspect"}
            else store,
        )
        parent = issue_number(args.parent, gh.repo)
        if args.command == "publish-decision-log":
            from decision_logs.journal import Journal
            from decision_logs.publication import Publisher

            # Publication bookkeeping never enters the capture adapter: an
            # otherwise successful publication must not create another batch.
            owner = (
                store.read_data()
                .get("parents", {})
                .get(f"{gh.repo.casefold()}#{parent}", {})
            )
            if owner.get("session") != identity:
                raise ValueError("decision publication requires the assigned parent")
            value = Publisher(
                Path.cwd(), Journal(store.directory, gh.repo, parent)
            ).publish(dry_run=args.dry_run, retry=args.retry)
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
            return 2 if value["error"] or value["blocked"] else 0
        if args.command == "review-inspect":
            import importlib

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
            inspection_module = importlib.import_module("review_inspection")
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
            print(json.dumps(value, ensure_ascii=False, sort_keys=True))
            return 0 if value["status"] == "captured" else 2
        capture_source = True
        if args.command in {"improvement", "improvement-publish"}:
            owner = (
                store.read_data()
                .get("parents", {})
                .get(f"{gh.repo.casefold()}#{parent}", {})
            )
            capture_source = (
                owner.get("status") == "active" and owner.get("session") == identity
            )
        capture_token = CURRENT.set(
            decision_capture.begin(args, store, gh.repo, parent)
            if capture_source
            else None
        )
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
                dry_run=getattr(args, "dry_run", False),
                retry=getattr(args, "retry", False),
            )
            code = (
                2
                if log_result["unavailable_logs"]
                or any(r.get("error") for r in log_result["logs"])
                else 0
            )
            log_result = decision_capture.complete(args, log_result, exit_code=code)
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
            from .manual_workers import run_worker

            code, worker_result = run_worker(
                gh,
                store,
                parent,
                nonempty(identity, "parent identity"),
                child=number(args.child),
                prompt=args.prompt_file,
                resume=args.resume,
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
            worker_result = decision_capture.complete(
                args, worker_result, exit_code=code
            )
            print(
                json.dumps(worker_result, ensure_ascii=False, indent=2, sort_keys=True)
            )
            return code
        operation = (
            nullcontext()
            if args.command
            in {"status", "reconcile", "basis-input", "basis-check", "inspect-worker"}
            else store.operation(f"{gh.repo.casefold()}#{parent}")
        )
        project_operation = (
            store.operation(f"project:{args.owner.casefold()}#{args.project}")
            if args.command in {"project-setup", "project-sync"}
            else nullcontext()
        )
        transaction = nullcontext() if args.command in workers else store.locked()
        from .improvements import lock_key

        improvement_operation = (
            store.operation(lock_key(gh.repo))
            if args.command
            in {"improvement", "improvement-publish", "improvement-decide"}
            else nullcontext()
        )
        with improvement_operation, operation, project_operation, transaction:
            if args.command in workers:
                from .development import worker_command

                worker_result = worker_command(
                    args, gh, parent, nonempty(identity, "parent identity"), store
                )
                worker_result = decision_capture.complete(args, worker_result)
                print(
                    json.dumps(
                        worker_result, ensure_ascii=False, indent=2, sort_keys=True
                    )
                )
                return 0
            session = ParentSession(gh, store, parent, identity)
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
            elif command == "note":
                result = session.note(read_text(args.body_file))
            elif command == "workflow-report":
                from .development import load, report

                result = report(session, load(Path(args.spec)))
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

                    result = release(session, read_object(args.spec))
                else:
                    result = session.lifecycle(
                        "ended" if args.ended else "released", read_text(args.note_file)
                    )
            elif command == "handoff":
                note = read_text(args.note_file)
                result = (
                    session.accept(note, args.recover)
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
                    read_text(args.restart_evidence_file)
                    if args.restart_evidence_file
                    else None,
                )
            elif command in {"stop", "resume"}:
                result = session.stop_or_resume(
                    number(args.child), read_object(args.spec), command == "resume"
                )
            elif command == "improvement":
                result = session.improvement(
                    number(args.child) if args.child is not None else None,
                    read_object(args.spec),
                    args.key,
                )
            elif command == "improvement-publish":
                from .improvements import publish

                result = publish(session, args.candidate, args.existing)
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
            elif command == "review-prepare":
                from .review_choice import selection as review_choice

                result = prepare(
                    session,
                    [number(n) for n in args.prs],
                    read_text(args.reason_file) if args.reason_file else None,
                    review_selection=review_choice(session, args),
                )
            elif command == "review-request":
                from .review_choice import selection as review_choice

                chosen_review = review_choice(session, args)
                if args.prs:
                    from .review_handoff import request as handoff_review

                    if not args.evidence_file:
                        raise ValueError(
                            "--prs requires --evidence-file with verified results"
                        )
                    result = handoff_review(
                        session,
                        args.prs,
                        read_text(args.instruction_file),
                        read_text(args.evidence_file),
                        read_text(args.reason_file) if args.reason_file else None,
                        review_selection=chosen_review,
                    )
                else:
                    result = transition(
                        session,
                        args.run_id,
                        "instructed",
                        read_text(args.instruction_file),
                        review_selection=chosen_review,
                    )
            else:
                result = transition(
                    session, args.run_id, args.state, read_text(args.evidence_file)
                )
        integration_failed = False
        if command == "improvement-decide":
            from .improvement_runtime import reconcile_stop

            # The decision is durable before stopping; do not hold the source
            # operation or registry lock while a runner terminates.
            result["stop"] = reconcile_stop(
                GitHub(gh.repo, store=store),
                store,
                parent,
                nonempty(identity, "recorder identity"),
                args.candidate,
                args.dispatcher_state_dir,
            )
            integration_failed = result["stop"]["state"] == "stopping"
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
                display = {"status": "error", "error": str(exc)}
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
        result = decision_capture.complete(
            args, result, exit_code=1 if integration_failed else 0
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if integration_failed else 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        decision_capture.failed(exc)
        # Do not echo arbitrary gh payloads/registry contents containing local IDs.
        print(f"parent-session: {exc}", file=sys.stderr)
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
        result = _main(argv)
        row["exit_code"] = result
        return result


if __name__ == "__main__":
    raise SystemExit(main())
