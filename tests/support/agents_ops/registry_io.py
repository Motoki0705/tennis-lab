"""Prove external executables can acquire the real registry flock."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def probe_code(lock: Path, output: Path) -> str:
    return f"""
import fcntl as _flock, json as _json, os as _os
_os.makedirs({str(lock.parent)!r}, exist_ok=True)
with open({str(lock)!r}, "a") as _stream:
    try:
        _flock.flock(_stream, _flock.LOCK_EX | _flock.LOCK_NB)
        _held = False
    except BlockingIOError:
        _held = True
with open({str(output)!r}, "a") as _stream:
    _stream.write(_json.dumps({{"held": _held, "pid": _os.getpid()}}) + "\\n")
if _held:
    raise RuntimeError("registry lock held during external I/O")
"""


def probe(lock: Path) -> None:
    code = f"""
import fcntl
with open({str(lock)!r}, "a") as stream:
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr


def instrument(executable: Path, lock: Path, output: Path) -> None:
    text = executable.read_text()
    marker = "from __future__ import annotations\n"
    offset = (
        text.index(marker) + len(marker) if marker in text else text.index("\n") + 1
    )
    executable.write_text(text[:offset] + probe_code(lock, output) + text[offset:])


def status_runner(path: Path, lock: Path, output: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\n"
        + probe_code(lock, output)
        + """
import sys
from pathlib import Path
args = sys.argv[1:]
directory = Path(args[args.index("--run-dir") + 1])
record_path = directory / "result.json"
record = _json.loads(record_path.read_text())
if args[0] == "stop":
    record.update(result="failed", cleanup_complete=True, turn_end_reason="stopped", _alive=False)
    record_path.write_text(_json.dumps(record))
print(_json.dumps({"schema_version": 1, "run_id": record["run_id"],
    "runner_alive": record.get("_alive", False), "agent_alive": record.get("_alive", False),
    "agent_group_alive": record.get("_alive", False)}))
"""
    )
    path.chmod(0o700)
    return path


def observations(output: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows and all(row["held"] is False for row in rows)
    return rows
