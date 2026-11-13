"""Socket-controlled fake launch with the real common runner's status reader."""

from __future__ import annotations

import importlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    if args[0] == "status":
        proc = subprocess.run(
            [os.environ["WAIT_REAL_RUNNER"], *args],
            capture_output=True,
            text=True,
            check=False,
        )
        # Acknowledgement proves the waiter reached its running observation.
        # The test can update the parent while this observation is paused.
        if address := os.environ.get("WAIT_STATUS_SOCKET"):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                try:
                    connection.connect(address)
                except (FileNotFoundError, ConnectionRefusedError):
                    pass  # The test closes the one-shot gate after acknowledgement.
                else:
                    connection.sendall((proc.stdout.strip() + "\n").encode())
                    assert connection.recv(1) == b"C"
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    boundary = importlib.import_module("fake_manual_worker_runner")
    with redirect_stdout(io.StringIO()):
        code: int = boundary.main()
    directory = Path(args[args.index("--run-dir") + 1])
    writer = importlib.import_module("parent.state").Store.write_json
    identity = importlib.import_module("lib.opsproc").process_identity(os.getpid())
    path = directory / "result.json"
    record = json.loads(path.read_text())
    record.update(runner=identity, started_at=time.time())
    record["turn_end_reason"] = os.environ.get(
        "WAIT_TERMINAL_REASON", record["turn_end_reason"]
    )
    terminal = dict(record)
    final = (directory / "final.md").read_bytes()
    (directory / "final.md").unlink()
    record.update(
        result="running",
        turn_end_reason=None,
        exit_code=None,
        cleanup_complete=False,
        finished_at=None,
    )
    writer(path, record)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.environ["WAIT_RUNNER_SOCKET"])
        connection.sendall(b"ready\n")
        assert connection.recv(1) == b"F"
    (directory / "final.md").write_bytes(final)
    terminal["finished_at"] = time.time()
    writer(path, terminal)
    print(json.dumps({key: terminal[key] for key in boundary.SUMMARY_KEYS}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
