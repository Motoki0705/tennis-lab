#!/usr/bin/env python3
"""Transition and validate issue-subagent workflow state."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import issue_task_commands as _commands  # noqa: E402
import issue_task_state as _state  # noqa: E402

load_state = _state.load_state
check = _commands.check
check_artifact = _commands.check_artifact
transition = _commands.transition
apply_feasibility_verdict = _commands.apply_feasibility_verdict
apply_preflight_verdict = _commands.apply_preflight_verdict
apply_test_verdict = _commands.apply_test_verdict
apply_seal_verdict = _commands.apply_seal_verdict
apply_return_review = _commands.apply_return_review
block_task = _commands.block_task
apply_validation_verdict = _commands.apply_validation_verdict
finalize_pr = _commands.finalize_pr
capture_pr_evidence = _commands.capture_pr_evidence
run_check = _commands.run_check


def non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be blank")
    return value.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    feasibility_parser = subparsers.add_parser("feasibility-verdict")
    feasibility_parser.add_argument("task_dir", type=Path)
    feasibility_parser.add_argument("verdict", choices=("PASS", "BLOCKED"))
    feasibility_parser.add_argument("--kind", choices=_state.BLOCK_KINDS)
    feasibility_parser.add_argument("--reason", type=non_blank)

    transition_parser = subparsers.add_parser("transition")
    transition_parser.add_argument("task_dir", type=Path)
    transition_parser.add_argument("phase", choices=("planning", "implementation", "validation"))

    artifact_parser = subparsers.add_parser("artifact-check")
    artifact_parser.add_argument("task_dir", type=Path)
    artifact_parser.add_argument("artifact", choices=tuple(_commands.ARTIFACT_PATHS))

    fingerprint_parser = subparsers.add_parser("candidate-fingerprint")
    fingerprint_parser.add_argument("task_dir", type=Path)
    fingerprint_parser.add_argument("--revision")

    run_check_parser = subparsers.add_parser("run-check")
    run_check_parser.add_argument("task_dir", type=Path)
    run_check_parser.add_argument("stage", choices=tuple(_commands.RESULT_PATHS))
    run_check_parser.add_argument("check_id", type=non_blank)

    preflight_parser = subparsers.add_parser("preflight-verdict")
    preflight_parser.add_argument("task_dir", type=Path)
    preflight_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    test_verdict_parser = subparsers.add_parser("test-verdict")
    test_verdict_parser.add_argument("task_dir", type=Path)
    test_verdict_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    seal_parser = subparsers.add_parser("seal-verdict")
    seal_parser.add_argument("task_dir", type=Path)
    seal_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    return_review_parser = subparsers.add_parser("return-review")
    return_review_parser.add_argument("task_dir", type=Path)
    return_review_parser.add_argument("action", choices=("implementation", "exploration"))
    return_review_parser.add_argument("--reason", type=non_blank, required=True)

    block_parser = subparsers.add_parser("block")
    block_parser.add_argument("task_dir", type=Path)
    block_parser.add_argument("kind", choices=_state.BLOCK_KINDS)
    block_parser.add_argument("--reason", type=non_blank, required=True)

    verdict_parser = subparsers.add_parser("verdict")
    verdict_parser.add_argument("task_dir", type=Path)
    verdict_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    capture_parser = subparsers.add_parser("capture-pr")
    capture_parser.add_argument("task_dir", type=Path)
    capture_parser.add_argument("--pr-number", type=int, required=True)

    finalize_parser = subparsers.add_parser("finalize-pr")
    finalize_parser.add_argument("task_dir", type=Path)
    finalize_parser.add_argument("--pr-number", type=int, required=True)
    finalize_parser.add_argument("--head-sha", type=non_blank, required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("task_dir", type=Path)
    return parser.parse_args()


def _print_errors(errors: list[str]) -> int:
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("ok")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "feasibility-verdict":
            apply_feasibility_verdict(
                args.task_dir,
                args.verdict,
                kind=args.kind,
                reason=args.reason,
            )
        elif args.command == "transition":
            transition(args.task_dir, args.phase)
        elif args.command == "artifact-check":
            return _print_errors(check_artifact(args.task_dir, args.artifact))
        elif args.command == "candidate-fingerprint":
            state = load_state(args.task_dir)
            if args.revision:
                print(_commands.compute_revision_fingerprint(args.task_dir, state, args.revision))
            else:
                print(_commands.compute_candidate_fingerprint(args.task_dir, state))
        elif args.command == "run-check":
            return run_check(args.task_dir, args.stage, args.check_id)
        elif args.command == "preflight-verdict":
            apply_preflight_verdict(args.task_dir, args.verdict)
        elif args.command == "test-verdict":
            apply_test_verdict(args.task_dir, args.verdict)
        elif args.command == "seal-verdict":
            apply_seal_verdict(args.task_dir, args.verdict)
        elif args.command == "return-review":
            apply_return_review(args.task_dir, args.action, args.reason)
        elif args.command == "block":
            block_task(args.task_dir, args.kind, args.reason)
        elif args.command == "verdict":
            apply_validation_verdict(args.task_dir, args.verdict)
        elif args.command == "capture-pr":
            capture_pr_evidence(args.task_dir, pr_number=args.pr_number)
        elif args.command == "finalize-pr":
            finalize_pr(
                args.task_dir,
                pr_number=args.pr_number,
                head_sha=args.head_sha,
            )
        else:
            return _print_errors(check(args.task_dir))
        return 0
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
