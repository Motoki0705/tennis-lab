"""Add isolated-test status/records to the existing out-of-sandbox MCP server.

The CPU execution tool remains absent until an OS-enforced backend is verified.
This separate entrypoint leaves retrieval, agent profiles and report CLI intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from .mcp_server import serve
from .repro_isolation import LinuxRunner, TestRunner, probe
from .reproduction import Reproduction
from .retrieval import (
    MAX_RESPONSE_BYTES,
    NOTICE,
    Retrieval,
    RetrievalError,
    encode,
    strict_json,
)


class ReproductionServer(Retrieval):
    def __init__(
        self,
        repo: Path,
        base_sha: str,
        repository: str,
        round_id: str,
        journal_dir: Path,
        ledger_path: Path,
        cpu_nodeids: list[str],
        *,
        ops_memory_sha: str | None = None,
        runner: TestRunner | None = None,
        pytest_lock: Path | None = None,
    ) -> None:
        super().__init__(
            repo,
            base_sha,
            repository,
            round_id,
            journal_dir,
            ops_memory_sha=ops_memory_sha,
        )
        self.reproduction = Reproduction(
            repo,
            base_sha,
            round_id,
            journal_dir / "reproduction",
            ledger_path,
            cpu_nodeids,
            runner=runner,
            pytest_lock=pytest_lock,
        )
        self.tools.update({tool.name: tool for tool in self.reproduction.tools()})

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        if name != "cpu_tests":
            retrieved: dict[str, Any] = super().call(name, arguments)
            return retrieved
        # Even a call to the unavailable/unadvertised CPU tool is recorded as
        # needs-reproduction. It never bypasses the capability gate.
        from .reproduction import now

        receipt = uuid.uuid4().hex
        response: dict[str, Any] = {
            "schema_version": 1,
            "receipt": receipt,
            "round_id": self.round_id,
            "retrieved_at": now(),
            "base_sha": self.base_sha,
            "notice": NOTICE,
            "sources": [{"kind": "request", "tool": name}],
            "pagination": {"has_more": False, "next": None, "omitted": False},
        }
        try:
            self.journal.append(
                {
                    "event": "requested",
                    "receipt": receipt,
                    "requested_at": response["retrieved_at"],
                    "tool": name,
                    "arguments": arguments,
                }
            )
            try:
                result = self.reproduction.request(arguments)
                response.update(ok=True, data=result.data, sources=result.sources)
                if len(encode(response)) > MAX_RESPONSE_BYTES:
                    raise RetrievalError(
                        "RESPONSE_LIMIT",
                        "run recorded; read cpu_test_log in smaller ranges",
                        sources=result.sources,
                    )
            except RetrievalError as exc:
                response.pop("data", None)
                response.update(
                    ok=False,
                    sources=exc.sources or response["sources"],
                    error={
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                    },
                )
            response["retrieved_at"] = now()
            self.journal.append(
                {
                    "event": "received",
                    "receipt": receipt,
                    "response_sha256": hashlib.sha256(encode(response)).hexdigest(),
                    "response": response,
                }
            )
        except (OSError, ValueError) as exc:
            response.pop("data", None)
            response.update(
                ok=False, error={"code": "RECORDING_FAILED", "message": str(exc)}
            )
        return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspection = subparsers.add_parser(
        "probe", help="fixed enforcement challenges in temporary scopes and namespaces"
    )
    inspection.add_argument("--directory", type=Path, required=True)
    server = subparsers.add_parser(
        "serve", help="retrieval plus guarded CPU-test records"
    )
    server.add_argument("--repo", type=Path, required=True)
    server.add_argument("--base-sha", required=True)
    server.add_argument("--repository", required=True)
    server.add_argument(
        "--round-id", required=True, help="same YYYYMMDD as the existing ledger"
    )
    server.add_argument("--journal-dir", type=Path, required=True)
    server.add_argument("--ledger", type=Path, required=True)
    server.add_argument(
        "--cpu-tests-file",
        type=Path,
        required=True,
        help="host-reviewed JSON: base_sha and nodeids",
    )
    server.add_argument("--ops-memory-sha")
    server.add_argument(
        "--isolation-proof",
        type=Path,
        help="private probe.json from a successful matching OS probe (maximum age: 24h)",
    )
    args = parser.parse_args()
    try:
        if args.command == "probe":
            result = probe(args.directory)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["verified"] else 1
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
        retrieval = ReproductionServer(
            args.repo,
            args.base_sha,
            args.repository,
            args.round_id,
            args.journal_dir,
            args.ledger,
            allowlist["nodeids"],
            ops_memory_sha=args.ops_memory_sha,
            runner=LinuxRunner(args.isolation_proof),
        )
        serve(retrieval, sys.stdin.buffer, sys.stdout.buffer)
    except (OSError, ValueError, RetrievalError) as exc:
        print(
            json.dumps(
                {
                    "error": "REPRODUCTION_CONFIGURATION_OR_STORAGE_ERROR",
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
