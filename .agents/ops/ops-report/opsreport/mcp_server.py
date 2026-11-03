"""Audit ledger adapter using the shared retrieval MCP transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from lib.review_tools.mcp_server import (  # noqa: F401
    MAX_REQUEST_BYTES,
    dispatch,
    serve,
)

from .audit_tools import AuditTools
from .repro_isolation import LinuxRunner, TestRunner
from .retrieval import Retrieval, RetrievalError, strict_json


def source(args: argparse.Namespace, *, runner: TestRunner | None = None) -> Retrieval:
    if args.isolation_proof is not None and args.cpu_tests_file is None:
        raise ValueError("isolation proof requires a host-approved CPU-test allowlist")
    options = {"ops_memory_sha": args.ops_memory_sha}
    if args.cpu_tests_file is None:
        return Retrieval(
            args.repo,
            args.base_sha,
            args.repository,
            args.round_id,
            args.journal_dir,
            **options,
        )
    if args.ledger_path is None:
        raise ValueError("CPU-test recording requires the pinned audit ledger")
    raw = args.cpu_tests_file.read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError("CPU allowlist exceeds byte limit")
    allowlist = strict_json(raw)
    if (
        not isinstance(allowlist, dict)
        or set(allowlist) != {"base_sha", "nodeids"}
        or allowlist["base_sha"] != args.base_sha
    ):
        raise ValueError("host CPU-test allowlist must match the fixed base SHA")
    from .reproduction_server import ReproductionServer

    return ReproductionServer(
        args.repo,
        args.base_sha,
        args.repository,
        args.round_id,
        args.journal_dir,
        args.ledger_path,
        allowlist["nodeids"],
        runner=runner if runner is not None else LinuxRunner(args.isolation_proof),
        **options,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--ops-memory-sha")
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--ending-path", type=Path)
    parser.add_argument("--max-threads", type=int)
    parser.add_argument("--cpu-tests-file", type=Path)
    parser.add_argument("--isolation-proof", type=Path)
    args = parser.parse_args()
    try:
        retrieval = source(args)
        server: Retrieval | AuditTools = retrieval
        state_args = (
            args.ledger_path,
            args.runtime_dir,
            args.input_path,
            args.ending_path,
            args.max_threads,
        )
        if any(value is not None for value in state_args):
            if any(value is None for value in state_args):
                raise ValueError(
                    "all ledger/runtime/input/ending/max-threads arguments are required together"
                )
            server = AuditTools(
                retrieval,
                ledger_path=args.ledger_path,
                runtime_dir=args.runtime_dir,
                input_path=args.input_path,
                ending_path=args.ending_path,
                max_threads=args.max_threads,
            )
        serve(cast(Retrieval, server), sys.stdin.buffer, sys.stdout.buffer)
    except (OSError, ValueError, RetrievalError) as exc:
        print(
            json.dumps(
                {"error": "SERVER_CONFIGURATION_OR_STORAGE_ERROR", "message": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
