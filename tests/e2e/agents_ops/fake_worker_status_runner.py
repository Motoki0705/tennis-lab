"""Full runner records and socket synchronization over the manual CLI fixture."""

from __future__ import annotations

import importlib
import io
import json
import os
import socket
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path


def main() -> int:
    boundary = importlib.import_module("fake_manual_worker_runner")
    store = importlib.import_module("parent.state").Store
    process_identity = importlib.import_module("lib.opsproc").process_identity
    if sys.argv[1] == "status":
        status: int = boundary.main()
        return status
    # Reuse the producer test's durable admission/lock/contract assertions.
    with redirect_stdout(io.StringIO()):
        code: int = boundary.main()
    args = sys.argv[1:]
    directory = Path(args[args.index("--run-dir") + 1])
    path = directory / "result.json"
    record = json.loads(path.read_text())
    record["contract"]["settings"] = {"agent": "codex"}
    record.update(
        run_dir=str(directory),
        session_root=str(directory),
        runner=process_identity(os.getpid()),
        process=None,
        started_at=time.time(),
    )
    terminal = dict(record)
    record.update(
        result="running", turn_end_reason=None, exit_code=None, cleanup_complete=False
    )
    store.write_json(
        directory / "session.json",
        {
            "schema_version": 1,
            "session_id": record["runtime"]["session_id"],
            "last_turn_id": None,
            "latest_run": str(directory),
        },
    )
    (directory / "events.jsonl").write_bytes(b"")
    store.write_json(path, record)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.environ["WORKER_STATUS_TEST_SOCKET"])
        connection.sendall(b"ready\n")
        assert connection.recv(1) == b"F"
    terminal["finished_at"] = time.time()
    store.write_json(path, terminal)
    print(json.dumps({key: terminal[key] for key in boundary.SUMMARY_KEYS}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
