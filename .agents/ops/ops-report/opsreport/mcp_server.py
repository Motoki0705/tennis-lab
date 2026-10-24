"""Audit ledger adapter using the shared retrieval MCP transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from lib.review_tools.mcp_server import dispatch, serve  # noqa: F401

from .audit_tools import AuditTools
from .retrieval import Retrieval, RetrievalError


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
    args = parser.parse_args()
    try:
        retrieval = Retrieval(
            args.repo,
            args.base_sha,
            args.repository,
            args.round_id,
            args.journal_dir,
            ops_memory_sha=args.ops_memory_sha,
        )
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
