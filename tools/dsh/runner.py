"""Execute one pinned dsh turn in an explicit linked worktree and record evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import uuid4

from tools.dsh import headless_config as config
from tools.dsh.accounting import usage_record
from tools.dsh.events import event_prefix, truncation_summary
from tools.dsh.home_policy import remember_home, verify_home
from tools.dsh.run_process import Interrupt, execute
from tools.dsh.run_state import (
    RunError,
    diff_summary,
    external_path,
    json_bytes,
    private_directory,
    private_write,
    read_key,
    redact,
    snapshot,
    validate_worktree,
)

ACTIVE = "TENNIS_LAB_DSH_ACTIVE"
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
EXIT_CODES = {
    "completed": 0,
    "api_error": 10,
    "invalid_ndjson": 11,
    "dsh_failed": 12,
    "aborted": 13,
    "turn_stopped": 13,
    "timeout": 124,
}


def now() -> str:
    return datetime.now(UTC).isoformat()


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # argparse echoes arbitrary argv. Use one safe, machine-readable error.
        raise RunError(
            "INVALID_ARGUMENTS",
            "Expected --task <Markdown file> --worktree <linked root> --mode read-only|workspace-write --timeout <positive finite seconds> [--session-id <id>] [--key-file <path>] [--dsh-home <external directory>].",
        )


def positive_timeout(value: str) -> float:
    try:
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid duration")
        return seconds
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "timeout must be positive and finite"
        ) from error


def arguments() -> argparse.Namespace:
    parser = Parser(prog="tools/dsh/run.sh", description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--timeout", type=positive_timeout, required=True)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path.home() / ".config/tennis-lab/deepseek-api-key",
    )
    parser.add_argument(
        "--dsh-home",
        type=Path,
        default=Path.home() / ".local/state/tennis-lab-agents/dsh-home",
    )
    return parser.parse_args()


def blank_summary() -> dict[str, Any]:
    return {
        "result": "rejected",
        "exit_code": 2,
        "dsh_exit_code": None,
        "session_id": None,
        "turn": None,
        "turn_end_reason": None,
        "records_dir": None,
        "final_path": None,
        "diff_summary": None,
        "usage_totals": None,
        "observed_token_estimate_range_usd": None,
        "usage_missing": True,
        "cost_complete": False,
        "task_success": None,
        **truncation_summary([]),
        "meaning": "Exit 0 means a completed dsh turn only. The parent must verify the diff and tests to establish task success.",
    }


def inspect_turn(
    events: list[dict[str, Any]],
    *,
    complete: bool,
    cwd: Path,
    requested_session: str | None,
    dsh_exit: int | None,
) -> dict[str, Any]:
    """Validate the event interval, independently of task acceptance or text."""
    result: dict[str, Any] = {
        "result": "invalid_ndjson",
        "session_id": None,
        "turn": None,
        "turn_end_reason": None,
    }
    sessions = [event for event in events if event["type"] == "session"]
    if len(sessions) != 1 or events[0] is not sessions[0]:
        return result
    session = sessions[0]
    identity = session.get("sessionId")
    if (
        not isinstance(identity, str)
        or SESSION_ID.fullmatch(identity) is None
        or session.get("cwd") != str(cwd)
        or requested_session is not None
        and identity != requested_session
    ):
        return result
    result["session_id"] = identity
    statuses = [event for event in events if event["type"] == "status"]
    turn_ends = [event for event in statuses if event.get("phase") == "turn_end"]
    if len(turn_ends) == 1:
        reason = turn_ends[0].get("reason")
        if (
            isinstance(reason, dict)
            and isinstance(reason.get("kind"), str)
            and reason["kind"]
            in {
                "completed",
                "error",
                "aborted",
                "blocked",
                "max-tokens",
                "interrupted",
                "forked",
            }
        ):
            result["turn_end_reason"] = reason["kind"]
    if not statuses or any(type(event.get("turn")) is not int for event in statuses):
        return result
    turns = {event["turn"] for event in statuses}
    if len(turns) != 1:
        return result
    result["turn"] = statuses[0]["turn"]
    finals = [event for event in events if event["type"] == "final"]
    if (
        not complete
        or len(finals) != 1
        or events[-1] is not finals[0]
        or not isinstance(finals[0].get("text"), str)
        or len(turn_ends) != 1
        or statuses[-1] is not turn_ends[0]
    ):
        return result
    # Error turns can end before turn_start/step_end (e.g. auth/provider failure).
    if result["turn_end_reason"] == "error":
        result["result"] = "api_error"
        return result
    if result["turn_end_reason"] == "aborted":
        result["result"] = "aborted"
        return result
    if result["turn_end_reason"] in {"blocked", "max-tokens", "interrupted", "forked"}:
        result["result"] = "turn_stopped"
        return result
    starts = [event["step"] for event in statuses if event.get("phase") == "step_start"]
    ends = [event["step"] for event in statuses if event.get("phase") == "step_end"]
    phases = [event.get("phase") for event in statuses]
    if (
        result["turn_end_reason"] != "completed"
        or not starts
        or starts != ends
        or starts != list(range(1, len(starts) + 1))
        or phases
        != [
            "turn_start",
            *[phase for _ in starts for phase in ("step_start", "step_end")],
            "turn_end",
        ]
        or any(event["type"] == "error" for event in events)
    ):
        return result
    result["result"] = "completed" if dsh_exit == 0 else "dsh_failed"
    return result


def read_registry(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {}
    try:
        info = path.lstat()
        if path.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("unsafe registry")
        registry = json.loads(path.read_bytes())
        if not isinstance(registry, dict) or not all(
            isinstance(identity, str)
            and SESSION_ID.fullmatch(identity)
            and isinstance(entry, dict)
            and isinstance(entry.get("cwd"), str)
            and entry.get("mode") in config.MODES
            and isinstance(entry.get("runs"), list)
            and all(
                isinstance(run, dict)
                and isinstance(run.get("run_id"), str)
                and (run.get("turn") is None or type(run["turn"]) is int)
                and isinstance(run.get("usage_totals"), dict)
                and type(run.get("usage_complete")) is bool
                for run in entry["runs"]
            )
            for identity, entry in registry.items()
        ):
            raise ValueError("invalid registry")
        return registry
    except (OSError, ValueError) as error:
        raise RunError(
            "INVALID_SESSION_STATE",
            "The private wrapper session registry is invalid; preserve the evidence and inspect it locally.",
        ) from error


def store_registry(path: Path, registry: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    private_write(temporary, json_bytes(registry))
    temporary.replace(path)


def inspect_home_config(
    command: list[str],
    *,
    worktree: Path,
    environment: dict[str, str],
    mode: str,
    interrupt: Interrupt,
) -> None:
    with io.BytesIO() as output, io.BytesIO() as diagnostic:
        exit_code, forced = execute(
            command + ["--dump-config"],
            cwd=worktree,
            environment=environment,
            task=b"",
            timeout=30,
            interrupt=interrupt,
            stdout=output,
            stderr=diagnostic,
            secrets=[],
        )
        if forced == "timeout":
            raise RunError(
                "PREFLIGHT_TIMEOUT",
                "The persistent-home configuration inspection exceeded its 30-second limit.",
            )
        if forced is not None or exit_code != 0 or diagnostic.getvalue().strip():
            raise config.ConfigError(
                "The persistent DSH_HOME configuration inspection failed or warned; preserve the private state and inspect the runtime."
            )
        try:
            config.inspect_dump(output.getvalue().decode("utf-8"), mode)
        except UnicodeError as error:
            raise config.ConfigError(
                "Invalid encoding in the persistent DSH_HOME configuration dump."
            ) from error


def inspect_fresh_config(
    *, worktree: Path, environment: dict[str, str], mode: str, interrupt: Interrupt
) -> dict[str, Any]:
    # The existing check_config implementation uses short subprocess.run calls.
    # Give the whole inspection its own process group so a wrapper interruption
    # also stops the inspection's runtime/CLI descendants.
    command = [
        sys.executable,
        "-B",
        str(config.DIRECTORY / "headless_config.py"),
        "--mode",
        mode,
    ]
    with io.BytesIO() as output, io.BytesIO() as diagnostic:
        exit_code, forced = execute(
            command,
            cwd=worktree,
            environment=environment,
            task=b"",
            timeout=30,
            interrupt=interrupt,
            stdout=output,
            stderr=diagnostic,
            secrets=[],
        )
        if forced == "timeout":
            raise RunError(
                "PREFLIGHT_TIMEOUT",
                "The fresh-home check_config inspection exceeded its 30-second limit.",
            )
        if forced is not None or exit_code != 0 or diagnostic.getvalue().strip():
            raise config.ConfigError(
                "The fresh-home check_config inspection failed or warned; inspect the pinned runtime and policy files."
            )
        try:
            value = json.loads(output.getvalue())
            if not isinstance(value, dict):
                raise ValueError("invalid config result")
            return value
        except ValueError as error:
            raise config.ConfigError(
                "Invalid check_config inspection result."
            ) from error


def run(options: argparse.Namespace, interrupt: Interrupt) -> dict[str, Any]:
    summary = blank_summary()
    worktree, main = validate_worktree(options.worktree)
    ignored = subprocess.run(
        ["git", "-C", str(worktree), "check-ignore", "-q", "--", ".dsh-runs/"],
        env={"PATH": os.environ.get("PATH", os.defpath)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=15,
    )
    if ignored.returncode != 0:
        raise RunError(
            "UNSAFE_RECORDS",
            "The target worktree must ignore /.dsh-runs/ before execution.",
        )
    root = private_directory(worktree / ".dsh-runs")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    records = root / run_id
    records.mkdir(mode=0o700)
    summary.update(
        run_id=run_id, records_dir=str(records), final_path=str(records / "final.md")
    )
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "cwd": str(worktree),
        "requested_session_id": options.session_id,
        "started_utc": now(),
        "timeout_seconds": options.timeout,
        "mode": options.mode,
    }
    events: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    home: Path | None = None
    registry: dict[str, Any] = {}
    lock: int | None = None
    secret = ""
    stream_complete = False
    try:
        if ACTIVE in os.environ or os.environ.get("DSH_SHELL") == "1":
            raise RunError(
                "RECURSIVE_CALL",
                "Calling the dsh wrapper from inside dsh is forbidden.",
            )
        mode = config.permission_mode(options.mode)
        key_path = external_path(options.key_file.expanduser(), main)
        # Reject missing/empty/unsafe key files before invoking even a keyless
        # CLI configuration inspection. The value is never added to os.environ.
        secret = read_key(key_path)
        before = snapshot(worktree, records, "before", [secret.encode()])
        home = external_path(options.dsh_home.expanduser(), main)
        private_directory(home)
        lock = os.open(
            home / ".runner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunError(
                "DSH_HOME_BUSY",
                "Another wrapper run owns this DSH_HOME; wait for it to finish.",
            ) from error
        verify_home(home)
        registry = read_registry(home / ".runner-sessions.json")
        if options.session_id is not None:
            previous = registry.get(options.session_id)
            if not isinstance(previous, dict):
                raise RunError(
                    "UNKNOWN_SESSION",
                    "Unknown session_id in this wrapper DSH_HOME; omit --session-id to start a new conversation.",
                )
            if previous.get("cwd") != str(worktree):
                raise RunError(
                    "SESSION_CWD_MISMATCH",
                    "This session belongs to another worktree; continue in the original root.",
                )
            if previous["mode"] != mode:
                raise RunError(
                    "SESSION_MODE_MISMATCH",
                    "Continue a session in its original safe mode, or omit --session-id to start a new conversation.",
                )
        if options.task.suffix.lower() != ".md" or not options.task.is_file():
            raise RunError(
                "INVALID_TASK", "Use an existing UTF-8 Markdown task file (.md)."
            )
        task = options.task.read_bytes()
        try:
            if not task.decode("utf-8").strip():
                raise ValueError("empty task")
        except (UnicodeError, ValueError) as error:
            raise RunError(
                "INVALID_TASK", "The Markdown task must contain nonempty UTF-8 text."
            ) from error
        metadata.update(
            task_sha256=hashlib.sha256(task).hexdigest(), task_bytes=len(task)
        )
        arguments = config.headless_arguments(mode)
        node = shutil.which("node")
        if node is None:
            raise RunError(
                "MISSING_NODE",
                "Install Node and the pinned runtime as described in tools/dsh/README.md.",
            )
        environment = config.headless_environment(
            mode,
            home,
            {
                name: os.environ[name]
                for name in ("PATH", "HOME", "LANG", "LC_ALL", "TERM")
                if name in os.environ
            },
        )
        temporary = private_directory(home / "tmp")
        environment.update(TMPDIR=str(temporary), **{ACTIVE: "1"})
        metadata["config"] = inspect_fresh_config(
            worktree=worktree, environment=environment, mode=mode, interrupt=interrupt
        )
        command = [node, str(config.RUNTIME), "exec", "--", *arguments]
        inspect_home_config(
            command,
            worktree=worktree,
            environment=environment,
            mode=mode,
            interrupt=interrupt,
        )
        metadata.update(remember_home(home))
        if options.session_id is not None:
            with io.BytesIO() as output, io.BytesIO() as diagnostic:
                state_exit, state_forced = execute(
                    [
                        node,
                        str(config.DIRECTORY / "session_state.mjs"),
                        str(home),
                        options.session_id,
                    ],
                    cwd=worktree,
                    environment=environment,
                    task=b"",
                    timeout=30,
                    interrupt=interrupt,
                    stdout=output,
                    stderr=diagnostic,
                    secrets=[],
                )
                if state_forced == "timeout":
                    raise RunError(
                        "PREFLIGHT_TIMEOUT",
                        "The persisted-session inspection exceeded its 30-second limit.",
                    )
                if state_exit != 0 or state_forced is not None or diagnostic.getvalue():
                    raise RunError(
                        "INVALID_SESSION_STATE",
                        "Cannot read this session using the pinned persistence backend; preserve its private logs and start a new session.",
                    )
                try:
                    state = json.loads(output.getvalue())
                except ValueError as error:
                    raise RunError(
                        "INVALID_SESSION_STATE",
                        "Invalid persisted session inspection result.",
                    ) from error
                if (
                    not isinstance(state, dict)
                    or state.get("cwd") != str(worktree)
                    or state.get("session_id") != options.session_id
                ):
                    raise RunError(
                        "SESSION_CWD_MISMATCH",
                        "The persisted session identity/cwd disagrees with the wrapper registry.",
                    )
                if (
                    state.get("mode") != mode
                    or state.get("approval") != "ask"
                    or state.get("preset") != mode
                ):
                    raise RunError(
                        "SESSION_MODE_MISMATCH",
                        "The persisted session permissions do not match this invocation. Continue with the original safe mode, or start a new session; permissions are never silently changed.",
                    )
                prior_turns = [
                    entry["turn"]
                    for entry in registry[options.session_id]["runs"]
                    if type(entry.get("turn")) is int
                ]
                if type(state.get("last_turn")) is not int or state["last_turn"] != max(
                    prior_turns, default=0
                ):
                    raise RunError(
                        "SESSION_TURN_MISMATCH",
                        "The persisted turn history disagrees with the recorded wrapper runs; preserve evidence and start a new session.",
                    )
                metadata["resumed_session"] = state
            command.extend(["--session-id", options.session_id])
        command.extend(["--json", "-"])
        metadata.update(command=command, dsh_home=str(home))
        if interrupt.signal is not None:
            summary.update(result="interrupted", exit_code=128 + interrupt.signal)
        else:
            # This is the only credential source; read at the last possible
            # moment, and inject into this child environment only.
            secret = read_key(key_path)
            environment["DEEPSEEK_API_KEY"] = secret
            for name in ("events.ndjson", "stderr.log"):
                private_write(records / name, b"")
            with (
                (records / "events.ndjson").open("wb") as output,
                (records / "stderr.log").open("wb") as diagnostic,
            ):
                dsh_exit, forced = execute(
                    command,
                    cwd=worktree,
                    environment=environment,
                    task=task,
                    timeout=options.timeout,
                    interrupt=interrupt,
                    stdout=output,
                    stderr=diagnostic,
                    secrets=[secret.encode()],
                )
            environment.pop("DEEPSEEK_API_KEY", None)
            summary["dsh_exit_code"] = dsh_exit
            event_data = (records / "events.ndjson").read_bytes()
            events, stream_complete = event_prefix(event_data)
            summary.update(
                inspect_turn(
                    events,
                    complete=stream_complete,
                    cwd=worktree,
                    requested_session=options.session_id,
                    dsh_exit=dsh_exit,
                )
            )
            expected_turn = metadata.get("resumed_session", {}).get("last_turn", 0) + 1
            if summary.get("turn") is not None and summary["turn"] != expected_turn:
                summary["result"] = "invalid_ndjson"
            if forced:
                summary["result"] = forced
            elif not event_data and dsh_exit != 0:
                summary["result"] = "dsh_failed"
            summary["exit_code"] = (
                128 + (interrupt.signal or signal.SIGINT)
                if summary["result"] == "interrupted"
                else EXIT_CODES[summary["result"]]
            )
    except (RunError, config.ConfigError) as error:
        summary.update(
            error_code=getattr(error, "code", "CONFIG_ERROR"), guidance=str(error)
        )
        if getattr(error, "code", None) == "PREFLIGHT_TIMEOUT":
            summary.update(result="timeout", exit_code=124)
    except (OSError, subprocess.SubprocessError):
        summary.update(
            result="execution_failed",
            exit_code=3,
            error_code="EXECUTION_FAILED",
            guidance="Cannot prepare or execute the local run; inspect paths, permissions and runtime installation.",
        )
    finally:
        if interrupt.signal is not None:
            summary.update(result="interrupted", exit_code=128 + interrupt.signal)
        final = next(
            (
                event["text"]
                for event in reversed(events)
                if event["type"] == "final" and isinstance(event.get("text"), str)
            ),
            "",
        )
        try:
            final_bytes = final.encode("utf-8")
        except UnicodeError:
            summary.update(
                result="invalid_ndjson",
                exit_code=11,
                error_code="INVALID_FINAL_ENCODING",
            )
            final_bytes = b""
        private_write(records / "final.md", final_bytes)
        for name in ("events.ndjson", "stderr.log"):
            if not (records / name).exists():
                private_write(records / name, b"")
        try:
            after = snapshot(worktree, records, "after", [secret.encode()])
            summary["diff_summary"] = diff_summary(before, after) if before else None
        except (OSError, RunError):
            summary.update(
                result="recording_failed", exit_code=3, error_code="GIT_EVIDENCE_FAILED"
            )
        usage = usage_record(
            events,
            scope="This run only; observed step_end usage. Historical session usage is excluded; input/cache/output buckets are disjoint.",
            stream_complete=stream_complete
            and summary["result"]
            in {"completed", "api_error", "aborted", "turn_stopped", "dsh_failed"}
            and len([event for event in events if event.get("phase") == "step_start"])
            == len([event for event in events if event.get("phase") == "step_end"]),
        )
        identity = summary["session_id"]
        usage.update(run_id=run_id, session_id=identity, turn=summary.get("turn"))
        usage["turns"] = [
            {
                "turn": turn,
                "usage": usage_record(
                    [event for event in events if event.get("turn") == turn],
                    scope="This run and turn only.",
                    stream_complete=usage["stream_complete"],
                ),
            }
            for turn in sorted(
                {
                    event["turn"]
                    for event in events
                    if event["type"] == "status" and type(event.get("turn")) is int
                }
            )
        ]
        summary.update(
            **truncation_summary(events),
            usage_totals=usage["totals"],
            observed_token_estimate_range_usd=usage[
                "observed_token_estimate_range_usd"
            ],
            usage_missing=not usage["usage_complete"],
        )
        if identity is not None and home is not None and lock is not None:
            previous = registry.get(
                identity, {"cwd": str(worktree), "mode": options.mode, "runs": []}
            )
            previous["runs"].append(
                {
                    "run_id": run_id,
                    "turn": summary.get("turn"),
                    "records_dir": str(records),
                    "result": summary["result"],
                    "usage_totals": usage["totals"],
                    "usage_complete": usage["usage_complete"],
                }
            )
            registry[identity] = previous
            try:
                store_registry(home / ".runner-sessions.json", registry)
            except OSError:
                summary.update(
                    result="recording_failed",
                    exit_code=3,
                    error_code="SESSION_REGISTRY_FAILED",
                )
            usage["session_runs"] = previous["runs"]
            usage["session"] = usage_record(
                [
                    {
                        "type": "status",
                        "phase": "step_end",
                        "turn": entry["turn"],
                        "step": index + 1,
                        "usage": entry["usage_totals"],
                    }
                    for index, entry in enumerate(previous["runs"])
                ],
                scope="All observed runs of this wrapper-owned session; run totals above exclude this history.",
                stream_complete=all(
                    entry["usage_complete"] for entry in previous["runs"]
                ),
            )
        private_write(records / "usage.json", json_bytes(usage))
        metadata.update(
            ended_utc=now(),
            ndjson_complete=stream_complete,
            interrupted_signal=interrupt.signal,
            **summary,
        )
        private_write(
            records / "metadata.json", json_bytes(metadata), [secret.encode()]
        )
        private_write(records / "summary.json", json_bytes(summary), [secret.encode()])
        if lock is not None:
            os.close(lock)
    return cast(
        dict[str, Any], json.loads(redact(json_bytes(summary), [secret.encode()]))
    )


def main() -> int:
    os.umask(0o077)
    interrupt = Interrupt()
    saved = {
        number: signal.signal(number, interrupt.receive)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        try:
            summary = run(arguments(), interrupt)
        except (RunError, config.ConfigError) as error:
            summary = {
                **blank_summary(),
                "error_code": getattr(error, "code", "CONFIG_ERROR"),
                "guidance": str(error),
            }
        except (OSError, subprocess.SubprocessError):
            summary = {
                **blank_summary(),
                "error_code": "PREFLIGHT_FAILED",
                "guidance": "Cannot prepare private records; check worktree paths and permissions.",
            }
        print(json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return int(summary["exit_code"])
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    sys.exit(main())
