"""Explicit parent orchestration. Writes only when an agent calls a subcommand."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from .agreement_comments import record_ignored
from .execution import execution_input, record_autonomous, validate_candidate
from .github import GitHub
from .policy import issue_number, nonempty, number, object_value
from .proposals import receive
from .session import ParentSession
from .snapshot import prepare, transition
from .state import Store, session_identity, state_directory


def read_text(path: str) -> str:
    return nonempty(Path(path).read_text(encoding="utf-8"), path)


def read_object(path: str) -> dict[str, Any]:
    return object_value(json.loads(read_text(path)))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repo", help="owner/repo; default: gh repo view")
    root.add_argument(
        "--parent",
        required=True,
        help="issue number or https://github.com/owner/repo/issues/N",
    )
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("register", "status", "checking", "reconcile", "sync"):
        commands.add_parser(command)
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
    release.add_argument("--note-file", required=True)
    handoff = commands.add_parser("handoff")
    handoff.add_argument("--accept", action="store_true")
    handoff.add_argument(
        "--recover",
        action="store_true",
        help="explicit takeover after inspecting an unexpectedly interrupted owner",
    )
    handoff.add_argument("--note-file", required=True)
    agree = commands.add_parser("agree")
    agree.add_argument("--spec", required=True)
    agree.add_argument("--report", type=int)
    agree.add_argument("--proposal")
    commands.add_parser("create-child").add_argument("--spec", required=True)
    receipt = commands.add_parser("receive-proposal")
    receipt.add_argument("--report", type=int, required=True)
    receipt.add_argument("--proposal", required=True)
    delegate = commands.add_parser("delegate")
    delegate.add_argument("--child", type=int, required=True)
    delegate.add_argument("--agent", choices=("claude", "codex"), required=True)
    delegate.add_argument("--branch", required=True)
    delegate.add_argument("--worktree", type=Path, required=True)
    delegate.add_argument("--dependencies", type=int, nargs="*")
    delegate.add_argument("--restart-evidence-file")
    result = commands.add_parser("record-result")
    result.add_argument("--child", type=int, required=True)
    result.add_argument("--evidence-file", required=True)
    for command in ("stop", "resume", "improvement"):
        sub = commands.add_parser(command)
        sub.add_argument("--child", type=int, required=True)
        sub.add_argument("--spec", required=True)
    review = commands.add_parser("review-prepare")
    review.add_argument("--prs", type=int, nargs="+", required=True)
    review.add_argument("--reason-file")
    request = commands.add_parser("review-request")
    request.add_argument("--run-id", required=True)
    request.add_argument("--instruction-file", required=True)
    state = commands.add_parser("review-state")
    state.add_argument("--run-id", required=True)
    state.add_argument(
        "--state",
        choices=("result_received", "fixes_verified", "complete"),
        required=True,
    )
    state.add_argument("--evidence-file", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "handoff" and args.recover and not args.accept:
            raise ValueError("--recover requires handoff --accept")
        # Missing identity is a configuration error before any network/state write.
        identity = (
            None
            if args.command in {"status", "basis-input", "basis-record", "basis-check"}
            else session_identity(dict(os.environ))
        )
        gh = GitHub(args.repo)
        parent = issue_number(args.parent, gh.repo)
        with Store(state_directory()).locked() as store:
            session = ParentSession(gh, store, parent, identity)
            command = args.command
            result: Any
            if command == "basis-input":
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
                result = session.register()
            elif command == "status":
                # Status works outside a live session; disclose no raw session identifiers.
                with suppress(ValueError):
                    session.session = session_identity(dict(os.environ))
                result = session.status()
            elif command == "checking":
                result = session.checking()
            elif command == "agree":
                result = session.agree(
                    read_object(args.spec), args.report, args.proposal
                )
            elif command == "release":
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
                result = {"progress": session.sync_progress()}
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
                result = session.improvement(number(args.child), read_object(args.spec))
            elif command == "record-result":
                result = session.record_result(
                    number(args.child), read_text(args.evidence_file)
                )
            elif command == "review-prepare":
                result = prepare(
                    session,
                    [number(n) for n in args.prs],
                    read_text(args.reason_file) if args.reason_file else None,
                )
            elif command == "review-request":
                result = transition(
                    session, args.run_id, "instructed", read_text(args.instruction_file)
                )
            else:
                result = transition(
                    session, args.run_id, args.state, read_text(args.evidence_file)
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        # Do not echo arbitrary gh payloads/registry contents containing local IDs.
        print(f"parent-session: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
