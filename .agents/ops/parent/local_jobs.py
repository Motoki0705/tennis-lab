"""Detached systemd runner and state helpers for long GPU-free local commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dispatcher.runtime import alive, process_identity
from lib.state_files import (
    atomic_state_writer,
    create_state_directory,
    open_state_directory,
)

from .state import state_directory, unique_object
from .worker_result import is_valid_local_job_id

SCHEMA_VERSION = 1
UNIT_PREFIX = "tennis-lab-local-job-"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
TERMINAL_STATES = ("done", "failed", "cancelled")
LOG_TAIL_BYTES = 32768
LOCAL_JOB_ENV_VAR = "PARENT_LOCAL_JOBS_DIR"
FORWARDED_ENV = (
    "PATH",
    "HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "VIRTUAL_ENV",
    "PYTHONDONTWRITEBYTECODE",
)
STRIPPED_SYSTEMD_ENV = frozenset(
    {
        "NOTIFY_SOCKET",
        "WATCHDOG_PID",
        "WATCHDOG_USEC",
        "LISTEN_FDS",
        "LISTEN_PID",
        "LISTEN_FDNAMES",
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "MAINPID",
        "SYSTEMD_EXEC_PID",
    }
)


class _SignalExit(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def notify_ready() -> None:
    """Signal readiness to systemd when running under Type=notify."""
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.sendto(
                b"READY=1", "\0" + address[1:] if address[0] == "@" else address
            )


def jobs_root(explicit: Path | None = None) -> Path:
    """Resolve the local-jobs storage root matching parent.state.state_directory()."""
    if explicit is not None:
        root = explicit
    elif configured := os.environ.get(LOCAL_JOB_ENV_VAR):
        root = Path(configured)
    else:
        root = state_directory() / "local-jobs"
    if not root.is_absolute():
        raise ValueError(f"{LOCAL_JOB_ENV_VAR} must be absolute")
    return root


def validate_name(name: str) -> str:
    if not isinstance(name, str) or ".." in name or not LABEL.fullmatch(name):
        raise ValueError(
            "invalid --name; use 1-80 chars of [A-Za-z0-9._-] starting with alphanumeric and without '..'"
        )
    return name


def validate_job_id(job_id: Any) -> str:
    if not is_valid_local_job_id(job_id):
        raise ValueError(f"unknown local job_id: {job_id}")
    job = str(job_id)
    if len(UNIT_PREFIX + job + ".service") > 256:
        raise ValueError(f"unknown local job_id: {job_id}")
    return job


def new_job_id(name: str) -> str:
    validate_name(name)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{secrets.token_hex(6)}_{name}"


def regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False
    if not stat.S_ISREG(mode):
        raise ValueError(f"local job path must be a regular file: {path}")
    return True


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with atomic_state_writer(path) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"{description}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description}: expected JSON object")
    return data


def _check_job_directory(root: Path, job_id: str) -> Path:
    if root.is_symlink():
        raise ValueError(f"local jobs root must not be a symlink: {root}")
    job_dir = root / job_id
    if job_dir.is_symlink():
        raise ValueError(f"local job directory must not be a symlink: {job_dir}")
    return job_dir


def read_meta(root: Path, job_id: str) -> dict[str, Any]:
    job_id = validate_job_id(job_id)
    job_dir = _check_job_directory(root, job_id)
    meta_path = job_dir / "meta.json"
    if not regular_file(meta_path):
        raise ValueError(f"unknown local job_id: {job_id}")
    meta = _read_json_object(meta_path, f"invalid local job metadata: {job_id}")
    if (
        type(meta.get("schema_version")) is not int
        or meta["schema_version"] != SCHEMA_VERSION
        or meta.get("job_id") != job_id
        or not isinstance(meta.get("name"), str)
        or not meta["name"]
        or not isinstance(meta.get("cwd"), str)
        or not meta["cwd"]
        or not isinstance(meta.get("command"), list)
        or not meta["command"]
        or any(not isinstance(item, str) for item in meta["command"])
        or not isinstance(meta.get("started_at"), str)
        or not meta["started_at"]
    ):
        raise ValueError(f"invalid local job metadata: {job_id}")
    return meta


def _valid_process_identity(proc: Any) -> bool:
    return (
        isinstance(proc, dict)
        and set(proc) == {"pid", "start_ticks", "boot_id"}
        and type(proc.get("pid")) is int
        and proc["pid"] > 0
        and isinstance(proc.get("start_ticks"), str)
        and proc["start_ticks"].isascii()
        and proc["start_ticks"].isdecimal()
        and isinstance(proc.get("boot_id"), str)
        and bool(proc["boot_id"])
    )


def read_running(root: Path, job_id: str) -> dict[str, Any] | None:
    job_dir = _check_job_directory(root, job_id)
    running_path = job_dir / "running.json"
    if not regular_file(running_path):
        return None
    record = _read_json_object(
        running_path, f"invalid local job running record: {job_id}"
    )
    proc = record.get("process")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != SCHEMA_VERSION
        or record.get("job_id") != job_id
        or (proc is not None and not _valid_process_identity(proc))
    ):
        raise ValueError(f"invalid local job running record: {job_id}")
    if proc is None:
        return None
    return dict(proc)


def read_completion(root: Path, job_id: str) -> dict[str, Any] | None:
    job_dir = _check_job_directory(root, job_id)
    completion_path = job_dir / "completion.json"
    if not regular_file(completion_path):
        return None
    record = _read_json_object(
        completion_path, f"invalid local job completion record: {job_id}"
    )
    state = record.get("state")
    exit_code = record.get("exit_code")
    duration = record.get("duration_seconds")
    name = record.get("name")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != SCHEMA_VERSION
        or record.get("job_id") != job_id
        or state not in TERMINAL_STATES
        or type(exit_code) is not int
        or (state == "done") != (exit_code == 0)
        or ("name" in record and (not isinstance(name, str) or not name))
        or (
            "duration_seconds" in record
            and duration is not None
            and (
                type(duration) not in (int, float)
                or not math.isfinite(duration)
                or duration < 0
            )
        )
    ):
        raise ValueError(f"invalid local job completion record: {job_id}")
    return record


def read_log_tail(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str | None:
    if not regular_file(path):
        return None
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        start = max(0, size - max_bytes)
        stream.seek(start)
        raw = stream.read()
    text = raw.decode("utf-8", errors="replace")
    if start > 0 and "\n" in text:
        text = text.split("\n", 1)[1]
    return text


def log_excerpt(log: str | None) -> str | None:
    if not log:
        return None
    lines = log.splitlines()
    errors = [line for line in lines if re.search(r"error|fail|traceback", line, re.I)]
    return "\n".join(line[-400:] for line in (errors or lines)[-12:])


def build_command(positional: list[str], shell: str | None) -> list[str]:
    if shell is not None and positional:
        raise ValueError("--shell and positional command are mutually exclusive")
    if shell is not None:
        if not shell.strip():
            raise ValueError("--shell command must not be empty")
        return ["bash", "-lc", shell]
    if not positional:
        raise ValueError("missing command; pass -- <cmd...> or --shell <cmd>")
    if not positional[0]:
        raise ValueError("command executable must not be empty")
    return list(positional)


def _resolve_cwd(raw: str | None) -> Path:
    cwd = Path(raw).resolve() if raw is not None else Path.cwd().resolve()
    if not cwd.is_dir():
        raise ValueError(f"working directory does not exist: {cwd}")
    return cwd


def _prepare_environment(caller_cwd: Path) -> tuple[str, dict[str, str]]:
    ops_dir = str(Path(__file__).resolve().parents[1])

    def absolutize_pythonpath(value: str) -> list[str]:
        items: list[str] = []
        for entry in value.split(os.pathsep):
            if not entry:
                continue
            path = Path(entry)
            resolved = str(
                path if path.is_absolute() else (caller_cwd / path).resolve()
            )
            if resolved not in items:
                items.append(resolved)
        return items

    caller_pythonpath = os.environ.get("PYTHONPATH", "")
    runner_entries = absolutize_pythonpath(caller_pythonpath)
    if ops_dir not in runner_entries:
        runner_entries.insert(0, ops_dir)
    runner_pythonpath = os.pathsep.join(runner_entries)

    child_env = {
        key: value
        for key, value in os.environ.items()
        if key not in STRIPPED_SYSTEMD_ENV
    }
    if caller_pythonpath:
        child_env["PYTHONPATH"] = os.pathsep.join(
            absolutize_pythonpath(caller_pythonpath)
        )
    return runner_pythonpath, child_env


def _record_start_failure(
    job_dir: Path,
    meta: dict[str, Any],
    exit_code: int,
    message: str,
) -> None:
    log_path = job_dir / "output.log"
    if not (job_dir / "output.log").exists():
        with suppress(OSError):
            fd = os.open(
                log_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(message.rstrip() + "\n")
    if not (job_dir / "completion.json").exists():
        with suppress(OSError, ValueError):
            write_json_atomic(
                job_dir / "completion.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": meta["job_id"],
                    "name": meta["name"],
                    "state": "failed",
                    "exit_code": exit_code,
                    "started_at": meta["started_at"],
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": 0.0,
                    "log_path": str(log_path),
                },
            )
    with suppress(OSError):
        (job_dir / "env.json").unlink()


def start_job(
    *,
    name: str,
    positional: list[str],
    shell: str | None = None,
    cwd: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Create job metadata and launch a detached transient systemd user service."""
    validate_name(name)
    command = build_command(positional, shell)
    work_dir = _resolve_cwd(cwd)
    storage_root = jobs_root(root)
    with open_state_directory(storage_root, create=True):
        pass
    job_id = new_job_id(name)
    job_dir = storage_root / job_id
    create_state_directory(job_dir, parents=False)

    unit = f"{UNIT_PREFIX}{job_id}"
    log_path = job_dir / "output.log"
    meta_path = job_dir / "meta.json"
    env_path = job_dir / "env.json"
    started_at = datetime.now(UTC).isoformat()
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "name": name,
        "cwd": str(work_dir),
        "command": command,
        "started_at": started_at,
        "unit": unit,
        "log_path": str(log_path),
        "runner_pid": None,
        "process": None,
    }
    runner_pythonpath, child_env = _prepare_environment(Path.cwd().resolve())
    write_json_atomic(meta_path, meta)
    write_json_atomic(env_path, {"schema_version": SCHEMA_VERSION, "env": child_env})

    env_options = [f"--setenv={key}" for key in FORWARDED_ENV if key in os.environ]
    for var in (LOCAL_JOB_ENV_VAR, "TRAINING_QUEUE_DIR"):
        if value := os.environ.get(var):
            env_options.append(f"--setenv={var}={Path(value).resolve()}")

    systemd_cmd = [
        "systemd-run",
        "--user",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        "--service-type=notify",
        "--property=TimeoutStartSec=30s",
        "--property=UMask=0077",
        f"--property=WorkingDirectory={work_dir}",
        f"--setenv=PYTHONPATH={runner_pythonpath}",
        *env_options,
        "--",
        sys.executable,
        "-m",
        "parent.local_jobs",
        "_run",
        str(storage_root),
        job_id,
    ]
    try:
        result = subprocess.run(
            systemd_cmd,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
    except FileNotFoundError as exc:
        message = f"systemd-run is unavailable: {exc}"
        _record_start_failure(job_dir, meta, 127, message)
        raise ValueError(message) from exc
    except subprocess.TimeoutExpired as exc:
        message = "systemd-run timed out starting local job service"
        _record_start_failure(job_dir, meta, 124, message)
        raise ValueError(message) from exc
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit {result.returncode}"
        )
        message = f"could not start local job service: {detail[:600]}"
        _record_start_failure(job_dir, meta, result.returncode, message)
        raise ValueError(message)

    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "name": name,
        "cwd": str(work_dir),
        "unit": unit,
        "log_path": str(log_path),
        "meta_path": str(meta_path),
        "started_at": started_at,
    }


def _load_child_env(job_dir: Path) -> dict[str, str]:
    env_path = job_dir / "env.json"
    if not regular_file(env_path):
        return {
            key: value
            for key, value in os.environ.items()
            if key not in STRIPPED_SYSTEMD_ENV
        }
    try:
        payload = _read_json_object(env_path, "invalid local job environment")
        raw_env = payload.get("env")
        if (
            type(payload.get("schema_version")) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(raw_env, dict)
            or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in raw_env.items()
            )
        ):
            raise ValueError("invalid local job environment schema")
        return dict(raw_env)
    finally:
        with suppress(OSError):
            env_path.unlink()


def run_job(root: Path, job_id: str) -> int:
    """Service entrypoint: publish liveness, notify systemd, execute, and record completion."""
    if not root.is_absolute():
        raise ValueError("local jobs root must be absolute")
    job_id = validate_job_id(job_id)
    job_dir = _check_job_directory(root, job_id)
    meta = read_meta(root, job_id)
    if regular_file(job_dir / "completion.json"):
        raise ValueError(f"local job already completed: {job_id}")
    existing_proc = read_running(root, job_id)
    if existing_proc is not None and alive(existing_proc):
        raise ValueError(f"local job is already running: {job_id}")
    cwd = Path(meta["cwd"])
    if not cwd.is_dir():
        raise ValueError(f"local job working directory does not exist: {cwd}")
    log_path = job_dir / "output.log"
    if meta.get("log_path") != str(log_path):
        raise ValueError(f"unexpected local job log_path in metadata: {job_id}")

    child_env = _load_child_env(job_dir)
    identity = process_identity(os.getpid())
    if identity is None:
        raise ValueError("could not determine runner process identity")
    write_json_atomic(
        job_dir / "running.json",
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "runner_pid": os.getpid(),
            "process": identity,
            "started_at": meta["started_at"],
        },
    )
    meta_updated = {**meta, "runner_pid": os.getpid(), "process": identity}
    write_json_atomic(job_dir / "meta.json", meta_updated)

    start_mono = time.monotonic()
    proc: subprocess.Popen[bytes] | None = None
    cancelled_sig: int | None = None
    recorded_exit: int | None = None
    previous_handlers: dict[signal.Signals, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal cancelled_sig
        cancelled_sig = signum
        if proc is not None and proc.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        raise _SignalExit(signum)

    log_fd = os.open(
        log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handle_signal)
        notify_ready()
        with os.fdopen(log_fd, "wb") as log_stream:
            log_fd = -1
            try:
                proc = subprocess.Popen(
                    meta["command"],
                    cwd=cwd,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except PermissionError as exc:
                log_stream.write(
                    f"local_jobs: failed to execute command: {exc}\n".encode()
                )
                log_stream.flush()
                recorded_exit = 126
            except OSError as exc:
                log_stream.write(
                    f"local_jobs: failed to execute command: {exc}\n".encode()
                )
                log_stream.flush()
                recorded_exit = 127
            else:
                try:
                    raw_code = proc.wait()
                except _SignalExit:
                    if proc.poll() is None:
                        try:
                            proc.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            with suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                else:
                    recorded_exit = 128 + (-raw_code) if raw_code < 0 else raw_code
    except _SignalExit:
        pass
    finally:
        if log_fd >= 0:
            os.close(log_fd)
        for saved_sig, handler in previous_handlers.items():
            signal.signal(saved_sig, handler)
        duration = round(max(0.0, time.monotonic() - start_mono), 3)
        finished_at = datetime.now(UTC).isoformat()
        if cancelled_sig is not None:
            state = "cancelled"
            exit_code = 128 + cancelled_sig
        elif recorded_exit is not None:
            exit_code = recorded_exit
            state = "done" if exit_code == 0 else "failed"
        else:
            state = "failed"
            exit_code = 1
        write_json_atomic(
            job_dir / "completion.json",
            {
                "schema_version": SCHEMA_VERSION,
                "job_id": job_id,
                "name": meta["name"],
                "state": state,
                "exit_code": exit_code,
                "started_at": meta["started_at"],
                "finished_at": finished_at,
                "duration_seconds": duration,
                "log_path": str(log_path),
            },
        )
    return 0


def job_status(job_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Read-only status inspection of a local job."""
    storage_root = jobs_root(root)
    job_id = validate_job_id(job_id)
    meta = read_meta(storage_root, job_id)
    job_dir = storage_root / job_id
    log_path = job_dir / "output.log"
    tail = read_log_tail(log_path)

    def completed_view(completion: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "name": completion.get("name") or meta["name"],
            "cwd": meta["cwd"],
            "command": meta["command"],
            "unit": meta.get("unit"),
            "started_at": completion.get("started_at", meta["started_at"]),
            "finished_at": completion.get("finished_at"),
            "duration_seconds": completion.get("duration_seconds"),
            "state": completion["state"],
            "exit_code": completion["exit_code"],
            "runner_alive": False,
            "log_path": str(log_path) if tail is not None else meta.get("log_path"),
            "log_excerpt": log_excerpt(tail),
        }

    completion = read_completion(storage_root, job_id)
    if completion is not None:
        return completed_view(completion)
    proc = read_running(storage_root, job_id)
    is_alive = proc is not None and alive(proc)
    if not is_alive:
        completion = read_completion(storage_root, job_id)
        if completion is not None:
            return completed_view(completion)
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "name": meta["name"],
        "cwd": meta["cwd"],
        "command": meta["command"],
        "unit": meta.get("unit"),
        "started_at": meta["started_at"],
        "finished_at": None,
        "duration_seconds": None,
        "state": "running" if is_alive else "lost",
        "exit_code": None,
        "runner_alive": is_alive,
        "log_path": str(log_path) if tail is not None else meta.get("log_path"),
        "log_excerpt": log_excerpt(tail),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parent.local_jobs",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    start_cmd = sub.add_parser("start", help="Launch a detached local command job.")
    start_cmd.add_argument("--name", required=True, help="Job label ([A-Za-z0-9._-]+).")
    start_cmd.add_argument("--cwd", help="Working directory (defaults to current CWD).")
    start_cmd.add_argument(
        "--shell",
        help="Shell command run via bash -lc (mutually exclusive with -- <cmd...>).",
    )
    start_cmd.add_argument("command", nargs="*", help="Command and arguments after --.")

    status_cmd = sub.add_parser("status", help="Inspect a local job's state.")
    status_cmd.add_argument("job_id", help="Local job ID returned by start.")

    run_cmd = sub.add_parser("_run", help="Internal systemd service wrapper.")
    run_cmd.add_argument("root", type=Path)
    run_cmd.add_argument("job_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.subcommand == "start":
            payload = start_job(
                name=args.name,
                positional=args.command,
                shell=args.shell,
                cwd=args.cwd,
            )
            print(json.dumps(payload, ensure_ascii=False))
            return 0
        if args.subcommand == "status":
            print(json.dumps(job_status(args.job_id), ensure_ascii=False))
            return 0
        if args.subcommand == "_run":
            return run_job(args.root, args.job_id)
    except (OSError, ValueError) as exc:
        print(f"local_jobs: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unexpected subcommand: {args.subcommand}")


if __name__ == "__main__":
    raise SystemExit(main())
