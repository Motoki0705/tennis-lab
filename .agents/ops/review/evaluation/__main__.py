"""Explicit review-evaluation CLI. Normal pytest never runs a comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.review_tools.retrieval import RetrievalError
from worker_logs.journal import atomic

from .catalog import MODELS, catalog, frozen_input, prepare, read, verify_snapshot


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repo", type=Path, required=True)
    init = commands.add_parser("init")
    init.add_argument("--state-dir", type=Path, required=True)
    bundle = commands.add_parser("prepare")
    bundle.add_argument("--state-dir", type=Path, required=True)
    bundle.add_argument("--repo", type=Path, required=True)
    bundle.add_argument("--case", type=int, required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--repo", type=Path, required=True)
    serve.add_argument("--bundle", type=Path, required=True)
    serve.add_argument("--journal", type=Path, required=True)
    select = commands.add_parser("record-selection")
    select.add_argument("--state-dir", type=Path, required=True)
    select.add_argument("--spec", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--repo", type=Path, required=True)
    run.add_argument("--case", type=int, required=True)
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--parent-instruction-file", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--additional", action="store_true")
    run.add_argument("--reason-file", type=Path)
    collect = commands.add_parser("collect")
    collect.add_argument("--state-dir", type=Path, required=True)
    collect.add_argument("--run-id", required=True)
    status = commands.add_parser("status")
    status.add_argument("--state-dir", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--state-dir", type=Path, required=True)
    report.add_argument("--parent-adjudications", type=Path)
    report.add_argument("--output", type=Path)
    template = commands.add_parser("adjudication-template")
    template.add_argument("--state-dir", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("audit-reuse")
    replay.add_argument("--parent-state-dir", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        value: dict[str, Any]
        if args.command == "serve":
            from lib.review_tools.mcp_server import serve

            from .transport import FrozenRetrieval

            serve(
                FrozenRetrieval(args.repo, args.bundle, args.journal),
                sys.stdin.buffer,
                sys.stdout.buffer,
            )
            return 0
        if args.command == "verify":
            cases = catalog()["cases"]
            for case in cases:
                frozen_input(case)
                verify_snapshot(args.repo, case)
            value = {
                "schema_version": 1,
                "verified_cases": [case["pr"] for case in cases],
                "agent_invocations": 0,
            }
        elif args.command == "audit-reuse":
            from .replay import historical_reuse

            value = historical_reuse(args.parent_state_dir)
            atomic(args.output, value)
        elif args.command == "prepare":
            from .execution import load

            load(args.state_dir)
            prepare(args.repo, args.state_dir, args.case)
            value = {"case": args.case, "prepared": True, "agent_invocations": 0}
        else:
            from . import execution

            if args.command == "init":
                execution.initialize(args.state_dir)
                value = {"initialized": True, "agent_invocations": 0}
            elif args.command == "record-selection":
                from .report import configure_selection

                configure_selection(args.state_dir, read(args.spec))
                value = {"selection_fixed": True, "agent_invocations": 0}
            elif args.command == "run":
                value = execution.execute(
                    args.repo,
                    args.state_dir,
                    args.case,
                    args.model,
                    args.parent_instruction_file.read_text(),
                    execute_approved=args.execute,
                    additional=args.additional,
                    reason=args.reason_file.read_text() if args.reason_file else None,
                )
            elif args.command == "collect":
                value = execution.collect(args.state_dir, args.run_id)
            elif args.command == "status":
                state = execution.load(args.state_dir)
                value = {
                    "runs": [
                        {
                            key: row[key]
                            for key in ("id", "case_id", "model", "role", "status")
                        }
                        for row in state["runs"]
                    ],
                    "budget": execution.forecast(
                        state, execution.observations(args.state_dir, state)
                    ),
                }
            elif args.command == "adjudication-template":
                from .catalog import DATA, digest
                from .metrics import candidate_table

                gold = read(DATA / "gold.json")
                state = execution.load(args.state_dir)
                rows = execution.observations(args.state_dir, state)
                value = {
                    "schema_version": 1,
                    "gold_sha256": digest(gold),
                    "judgments": [
                        {
                            "run_id": row["run_id"],
                            "output_sha256": row["output_sha256"],
                            "adjudicator": "pending",
                            "matches": [],
                            "other_findings": [],
                        }
                        for row in rows
                    ],
                    "candidate_matches": candidate_table(rows, gold),
                }
                atomic(args.output, value)
            else:
                from .report import report

                value = report(
                    args.state_dir,
                    read(args.parent_adjudications)
                    if args.parent_adjudications
                    else None,
                )
                if args.output:
                    atomic(args.output, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        StopIteration,
        subprocess.SubprocessError,
        RetrievalError,
    ) as exc:
        print(f"review-evaluation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
