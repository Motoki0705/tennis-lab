"""stdio MCP entrypoint for the audit's allowlisted, read-only retrieval tools.

Run outside the agent sandbox: python -m opsreport.mcp_server --help.
The caller supplies immutable identities and an external private journal directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

from .retrieval import NOTICE, Retrieval, RetrievalError, encode, strict_json

MAX_REQUEST_BYTES = 64 * 1024


def dispatch(retrieval: Retrieval, request: Any) -> dict[str, Any] | None:
    request_id = request.get("id") if isinstance(request, dict) else None

    def error(code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return error(-32600, "Invalid JSON-RPC request")
    method = request.get("method")
    if "id" not in request:
        # Notifications cannot perform retrievals or change the server configuration.
        return None
    params = request.get("params", {})
    if not isinstance(params, dict):
        return error(-32602, "params must be an object")
    result: dict[str, Any]
    if method == "initialize":
        version = params.get("protocolVersion")
        if not isinstance(version, str) or len(version) > 64:
            return error(-32602, "protocolVersion must be a string")
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ops-report-retrieval", "version": "1.0"},
            "instructions": NOTICE,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        if set(params) - {"cursor", "_meta"} or params.get("cursor") is not None:
            return error(-32602, "tools/list has no cursor or other parameters")
        result = {"tools": [tool.public() for tool in retrieval.tools.values()]}
    elif method == "tools/call":
        if (
            set(params) - {"name", "arguments", "_meta"}
            or not isinstance(params.get("name"), str)
            or len(params["name"]) > 100
        ):
            return error(-32602, "only a tool name and its arguments are accepted")
        response = retrieval.call(params["name"], params.get("arguments", {}))
        result = {
            "content": [{"type": "text", "text": encode(response).decode()}],
            "isError": not response["ok"],
        }
    else:
        return error(-32601, "Method is not allowed")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(retrieval: Retrieval, source: BinaryIO, target: BinaryIO) -> None:
    while raw := source.readline(MAX_REQUEST_BYTES + 1):
        if len(raw) > MAX_REQUEST_BYTES:
            while not raw.endswith(b"\n"):
                raw = source.readline(MAX_REQUEST_BYTES + 1)
                if not raw:
                    break
            response: dict[str, Any] | None = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Request exceeds byte limit"},
            }
        else:
            try:
                request = strict_json(raw)
            except (ValueError, UnicodeError):
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Invalid or duplicate-key JSON",
                    },
                }
            else:
                response = dispatch(retrieval, request)
        if response is not None:
            target.write(encode(response) + b"\n")
            target.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--repository", required=True, help="owner/name on github.com")
    parser.add_argument("--round-id", required=True)
    parser.add_argument(
        "--journal-dir",
        type=Path,
        required=True,
        help="private directory outside the agent checkout; append-only retrieval.jsonl",
    )
    parser.add_argument(
        "--ops-memory-sha",
        help="separately pinned dedicated memory commit; default is base-sha",
    )
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
        serve(retrieval, sys.stdin.buffer, sys.stdout.buffer)
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
