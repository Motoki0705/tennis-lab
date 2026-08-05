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
transition = _commands.transition
apply_feasibility_verdict = _commands.apply_feasibility_verdict
apply_preflight_verdict = _commands.apply_preflight_verdict
apply_test_verdict = _commands.apply_test_verdict
apply_return_review = _commands.apply_return_review
block_task = _commands.block_task
apply_validation_verdict = _commands.apply_validation_verdict


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
    transition_parser.add_argument("phase", choices=_state.PHASES[2:])

    preflight_parser = subparsers.add_parser("preflight-verdict")
    preflight_parser.add_argument("task_dir", type=Path)
    preflight_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    test_verdict_parser = subparsers.add_parser("test-verdict")
    test_verdict_parser.add_argument("task_dir", type=Path)
    test_verdict_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    return_review_parser = subparsers.add_parser("return-review")
    return_review_parser.add_argument("task_dir", type=Path)
    return_review_parser.add_argument(
        "action",
        choices=("implementation", "exploration"),
    )
    return_review_parser.add_argument("--reason", type=non_blank, required=True)

    block_parser = subparsers.add_parser("block")
    block_parser.add_argument("task_dir", type=Path)
    block_parser.add_argument("kind", choices=_state.BLOCK_KINDS)
    block_parser.add_argument("--reason", type=non_blank, required=True)

    verdict_parser = subparsers.add_parser("verdict")
    verdict_parser.add_argument("task_dir", type=Path)
    verdict_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("task_dir", type=Path)

    return parser.parse_args()


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
        elif args.command == "preflight-verdict":
            apply_preflight_verdict(args.task_dir, args.verdict)
        elif args.command == "test-verdict":
            apply_test_verdict(args.task_dir, args.verdict)
        elif args.command == "return-review":
            apply_return_review(args.task_dir, args.action, args.reason)
        elif args.command == "block":
            block_task(args.task_dir, args.kind, args.reason)
        elif args.command == "verdict":
            apply_validation_verdict(args.task_dir, args.verdict)
        else:
            errors = check(args.task_dir)
            if errors:
                for error in errors:
                    print(f"error: {error}", file=sys.stderr)
                return 1
            print("ok")
        return 0
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
