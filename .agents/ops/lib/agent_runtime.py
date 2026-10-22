"""Persistent headless CLI turns; assignment and task acceptance belong to callers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from agent_events import (
    Events,
    EvidenceError,
    codex_checkpoint,
    interrupted_codex_turn,
    parse_json,
)
from agent_settings import command, resolve
from opsproc import (
    _live_group_members,
    _terminate_group,
    enable_child_subreaper,
    process_identity,
    terminate_descendants,
)

VERSION = 1
IDENTITY_ENV = {
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_PARENT_SESSION_ID",
    "CLAUDE_CODE_AGENT_ID",
    "CLAUDECODE",
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path) -> dict[str, Any]:
    value = parse_json(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != VERSION
    ):
        raise ValueError("invalid or unsupported runner record")
    return value


def public_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        k: result[k]
        for k in (
            "schema_version",
            "run_id",
            "result",
            "turn_end_reason",
            "exit_code",
            "task_success",
        )
    }


def branch_at(cwd: Path, required: bool) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode or proc.stdout.strip() != "true":
        if required:
            raise ValueError("managed tasks require a git worktree")
        return None
    proc = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise ValueError("failed to read worktree branch")
    return proc.stdout.strip() if proc.returncode == 0 else "(detached)"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action", nargs="?", choices=("run", "resume", "stop", "status"), default="run"
    )
    group = result.add_mutually_exclusive_group()
    group.add_argument("--agent", choices=("claude", "codex"))
    group.add_argument("--role", choices=("parent", "implementation"))
    result.add_argument("--cwd", type=Path)
    result.add_argument("--prompt-file", help="UTF-8 prompt path, or - for stdin")
    result.add_argument("--out", type=Path)
    result.add_argument("--mode", choices=("read-only", "write"), default="read-only")
    result.add_argument("--profile")
    result.add_argument("--model")
    result.add_argument("--effort")
    result.add_argument("--task-id")
    result.add_argument("--branch")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--resume-from", type=Path)
    result.add_argument("--cleanup-confirmation", type=Path)
    result.add_argument("--log-dir", type=Path)
    result.add_argument(
        "--config-file", type=Path, default=Path(__file__).with_name("agents.conf")
    )
    result.add_argument("--timeout", type=float)
    return result


def capture(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    run_dir: Path,
    result: dict[str, Any],
    save: Any,
    events: Events,
    timeout: float | None,
) -> tuple[int, str | None]:
    stop_signal = [0]

    def stop(signum: int, frame: Any) -> None:
        stop_signal[0] = signum

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    proc = None
    try:
        with (
            (run_dir / "prompt.md").open("rb") as prompt,
            (run_dir / "events.jsonl").open("wb") as raw,
            (run_dir / "stderr.log").open("wb") as stderr,
        ):
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdin=prompt,
                stdout=subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
            result["process"] = process_identity(proc.pid)
            save()
            assert proc.stdout is not None
            pending = b""
            started = time.monotonic()
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    if stop_signal[0]:
                        return 128 + stop_signal[0], "stopped"
                    if timeout is not None and time.monotonic() - started >= timeout:
                        return 124, "timeout"
                    ready = selector.select(0.1)
                    if ready:
                        chunk = os.read(proc.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        raw.write(chunk)
                        raw.flush()
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if line.strip():
                                previous = events.session
                                events.feed(line)
                                if previous != events.session:
                                    save()
                    elif proc.poll() is not None:
                        break
                if pending.strip():
                    events.feed(pending)
            # EOF alone is not process completion: a CLI may close stdout early.
            while proc.poll() is None:
                if int(stop_signal[0]):
                    return 128 + stop_signal[0], "stopped"
                if timeout is not None and time.monotonic() - started >= timeout:
                    return 124, "timeout"
                time.sleep(0.05)
            return proc.returncode, None
    finally:
        try:
            if proc is not None:
                result["terminated_pids"] = _terminate_group(proc, proc.pid, 1.0)
                result["cli_exit_code"] = proc.returncode
                result["terminated_pids"].extend(terminate_descendants(1.0))
                result["cleanup_complete"] = True
                if proc.stdout is not None:
                    proc.stdout.close()
            else:
                result["cleanup_complete"] = True
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cleanup_confirmation is not None:
        raise ValueError("--cleanup-confirmation requires stop")
    if (
        args.cwd is None
        or args.prompt_file is None
        or (args.agent is None and args.role is None)
    ):
        raise ValueError("--agent/--role, --cwd and --prompt-file are required")
    if args.run_dir is None and args.out is None:
        raise ValueError("--run-dir or --out is required")
    if args.task_id is not None and not args.task_id.strip():
        raise ValueError("--task-id must be nonempty")
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        raise ValueError("--timeout must be finite and positive")
    if args.role and args.profile is not None:
        raise ValueError("--role cannot be combined with --profile")
    if args.action == "resume" and (args.resume_from is None or not args.task_id):
        raise ValueError("resume requires --resume-from and --task-id")
    if args.action != "resume" and args.resume_from is not None:
        raise ValueError("--resume-from requires resume")
    cwd = args.cwd.resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("--cwd is not a directory")
    prompt = (
        sys.stdin.buffer.read()
        if args.prompt_file == "-"
        else Path(args.prompt_file).read_bytes()
    )
    prompt.decode("utf-8")
    previous = (
        read_json(args.resume_from.resolve() / "result.json")
        if args.resume_from
        else None
    )
    if previous:
        for key in ("model", "effort"):
            if getattr(args, key) is None:
                setattr(
                    args, key, previous["contract"]["settings"]["overrides"].get(key)
                )
    agent = args.agent or ("claude" if args.role == "parent" else "codex")
    settings = resolve(
        args.config_file, agent, args.profile, args.model, args.effort, args.mode
    )
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
    branch = branch_at(cwd, bool(args.task_id))
    if args.branch is not None and args.branch != branch:
        raise ValueError("selected branch differs from the worktree branch")
    contract = {
        "task_id": args.task_id or hashlib.sha256(prompt).hexdigest(),
        "cwd": str(cwd),
        "branch": branch,
        "detached_head": None,
        "mode": args.mode,
        "settings": settings,
        "runtime_home": str(home) if agent == "codex" else str(Path.home() / ".claude"),
    }
    if branch == "(detached)":
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode:
            raise ValueError("failed to read detached worktree HEAD")
        contract["detached_head"] = head.stdout.strip()
    if previous and contract != previous["contract"]:
        raise ValueError(
            "resume task/worktree/branch/settings differ from the saved contract"
        )
    run_id = str(uuid.uuid4())
    log_dir = (
        args.log_dir
        or Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        / "tennis-lab-agents/logs"
    )
    run_dir = (args.run_dir or log_dir / run_id).resolve()
    root = Path(previous["session_root"]) if previous else run_dir
    if previous and run_dir == Path(previous["run_dir"]):
        raise ValueError("resume needs a new --run-dir")
    out = args.out.absolute() if args.out else run_dir / "final.md"
    if args.prompt_file != "-" and out.resolve() == Path(args.prompt_file).resolve():
        raise ValueError("prompt and output paths must differ")
    if out.resolve() == args.config_file.resolve():
        raise ValueError("config and output paths must differ")
    if out.parent.resolve() == run_dir and out.name != "final.md":
        raise ValueError("output inside --run-dir must be named final.md")
    final = run_dir / "final.md"
    if (
        previous
        and out.resolve() != final
        and any(
            out.resolve().is_relative_to(directory)
            for directory in (root, Path(previous["run_dir"]))
        )
    ):
        raise ValueError("output must not overwrite previous session records")
    if previous is None:
        run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    with (root / "session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("session already has a live runner") from exc
        transcript_before = None
        if previous:
            state = read_json(root / "session.json")
            if state["latest_run"] != str(args.resume_from.resolve()):
                raise ValueError("resume-from is not the latest session run")
            if previous["result"] == "running":
                raise ValueError(
                    "previous run is still running or unreconciled; inspect status/stop first"
                )
            identity = previous.get("process")
            if not previous.get("cleanup_complete") or (
                identity is not None
                and (live(identity) or _live_group_members(identity["pid"]))
            ):
                raise ValueError(
                    "previous agent cleanup is unconfirmed or processes remain; reconcile before resume"
                )
            if not state["session_id"]:
                raise ValueError("previous runtime did not return a resumable session")
            if previous.get("runner_lost"):
                confirmation = previous.get("cleanup_confirmation")
                if not isinstance(confirmation, dict):
                    raise ValueError(
                        "previous supervisor cleanup confirmation is missing"
                    )
                validate_cleanup_confirmation(confirmation, previous)
            if previous.get("recovery_error"):
                raise ValueError(
                    "previous turn evidence is unconfirmed; inspect stop recovery"
                )
            if agent == "codex":
                transcript_before = codex_checkpoint(home, state["session_id"])
                if (
                    state["last_turn_id"] is not None
                    and transcript_before["turn_id"] != state["last_turn_id"]
                ):
                    raise ValueError("runtime session changed outside this runner")
            run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        else:
            state = {
                "schema_version": VERSION,
                "session_id": None,
                "last_turn_id": None,
            }
        state["latest_run"] = str(run_dir)
        events = Events(agent, settings["selected"], state["session_id"])
        previous_turn = state["last_turn_id"]
        result: dict[str, Any] = {
            "schema_version": VERSION,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "session_root": str(root),
            "previous_run": str(args.resume_from.resolve()) if previous else None,
            "transcript_before": transcript_before,
            "contract": contract,
            "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
            "final_path": str(final),
            "output_path": str(out),
            "result": "running",
            "turn_end_reason": None,
            "exit_code": None,
            "task_success": None,
            "cli_exit_code": None,
            "runner": process_identity(os.getpid()),
            "process": None,
            "terminated_pids": [],
            "cleanup_complete": False,
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
        }

        def save() -> None:
            result["runtime"] = events.snapshot()
            if events.session is not None:
                state["session_id"] = events.session
            state["last_turn_id"] = events.turn or previous_turn
            write_json(run_dir / "result.json", result)
            write_json(root / "session.json", state)

        (run_dir / "prompt.md").write_bytes(prompt)
        save()
        env = {k: v for k, v in os.environ.items() if k not in IDENTITY_ENV}
        env.update(
            AGENT_RUNTIME_RUN_DIR=str(run_dir),
            AGENT_RUNTIME_TASK_ID=contract["task_id"],
        )
        status: int = 1
        reason: str | None = "invalid_output"
        try:
            enable_child_subreaper()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.unlink(missing_ok=True)
            cmd = command(settings, args.mode, cwd, final, state["session_id"])
            write_json(
                run_dir / "invocation.json",
                {
                    "schema_version": VERSION,
                    "argv": cmd,
                    "removed_environment_keys": sorted(
                        IDENTITY_ENV & os.environ.keys()
                    ),
                },
            )
            status, reason = capture(
                cmd, cwd, env, run_dir, result, save, events, args.timeout
            )
            if reason is None:
                reason = "cli_error" if status else "invalid_output"
                if events.failed:
                    raw_errors = json.dumps(events.errors).lower()
                    reason = (
                        "usage_limit"
                        if any(
                            s in raw_errors
                            for s in (
                                "quota",
                                "usage limit",
                                "rate_limit",
                                "rate limit",
                            )
                        )
                        else "runtime_error"
                    )
                if status == 0 and not events.failed:
                    events.validate(final)
                    reason = "metadata_error"
                    events.evidence(home, cwd, previous_turn)
                    if result["terminated_pids"]:
                        reason, status = "descendants_reaped", 1
                    else:
                        if out != final:
                            out.write_bytes(final.read_bytes())
                        reason = "completed"
                if reason != "completed" and status == 0:
                    status = 1
        except FileNotFoundError:
            status, reason = 127, "startup_failed"
            result["error"] = "required executable or runtime file is missing"
        except EvidenceError as exc:
            status, reason = 1, "metadata_error"
            result["error"] = str(exc)
        except (ValueError, OSError, RuntimeError) as exc:
            status = 1
            result["error"] = str(exc)
        if result["process"] is not None and not result["cleanup_complete"]:
            status, reason = 1, "cleanup_failed"
        # Keep whatever runtime evidence exists even on interruption/CLI failure.
        if reason != "completed" and events.session:
            try:
                events.evidence(home, cwd, previous_turn)
            except (ValueError, OSError) as exc:
                result["evidence_error"] = str(exc)
        result.update(
            result="completed" if reason == "completed" else "failed",
            turn_end_reason=reason,
            exit_code=status if status >= 0 else 128 - status,
            finished_at=time.time(),
        )
        save()
        print(
            f"agent_exec: agent={agent} mode={args.mode}"
            + (f" profile={args.profile}" if args.profile else "")
            + f" model={settings['selected']['model']} effort={settings['selected']['effort']} reason={reason}",
            file=sys.stderr,
        )
        if result["error"]:
            message = (
                "empty final message"
                if "empty final message" in result["error"]
                else "inspect local result.json and stderr.log for details"
            )
            print("agent_exec: " + message, file=sys.stderr)
        return result


def live(identity: dict[str, Any] | None) -> bool:
    return identity is not None and process_identity(identity["pid"]) == identity


def recover_lost_turn(result: dict[str, Any], state: dict[str, Any]) -> None:
    contract = result["contract"]
    agent = contract["settings"]["agent"]
    saved = result["runtime"]
    events = Events(
        agent,
        contract["settings"]["selected"],
        saved["requested_session_id"] or saved["session_id"] or state["session_id"],
    )
    run_dir = Path(result["run_dir"])
    started_turns = 0
    for line in (run_dir / "events.jsonl").read_bytes().splitlines():
        if line.strip():
            events.feed(line)
            if parse_json(line).get("type") == "turn.started":
                started_turns += 1
    if (
        not events.started
        or not events.session
        or state["session_id"] not in (None, events.session)
    ):
        raise ValueError("interrupted run has no matching runtime session start")
    if agent == "codex":
        if started_turns > 1:
            raise ValueError("interrupted Codex run has multiple turn start events")
        before = result.get("transcript_before")
        if result["previous_run"] is not None and not isinstance(before, dict):
            raise ValueError("interrupted continuation has no transcript checkpoint")
        prompt = (run_dir / "prompt.md").read_bytes()
        if hashlib.sha256(prompt).hexdigest() != result["prompt_sha256"]:
            raise ValueError("saved prompt differs from the interrupted run input")
        home = Path(contract["runtime_home"])
        turn = interrupted_codex_turn(
            home, events.session, before, prompt.decode("utf-8")
        )
        if events.turn not in {None, turn} or saved["turn_id"] not in {None, turn}:
            raise ValueError("interrupted Codex event/transcript turn IDs disagree")
        previous_turn = before["turn_id"] if before else None
        if state["last_turn_id"] not in {previous_turn, turn}:
            raise ValueError("saved session turn differs from the interrupted run")
        events.turn = turn
        events.evidence(home, Path(contract["cwd"]), previous_turn)
    result["runtime"] = events.snapshot()
    state.update(
        session_id=events.session, last_turn_id=events.turn or state["last_turn_id"]
    )


def validate_cleanup_confirmation(
    confirmation: dict[str, Any], result: dict[str, Any]
) -> None:
    if (
        type(confirmation.get("schema_version")) is not int
        or confirmation["schema_version"] != VERSION
        or set(confirmation)
        != {
            "schema_version",
            "run_id",
            "runner",
            "process",
            "cleanup_complete",
            "evidence",
        }
        or any(
            confirmation[key] != result[key] for key in ("run_id", "runner", "process")
        )
        or confirmation["cleanup_complete"] is not True
        or not isinstance(confirmation["evidence"], str)
        or not confirmation["evidence"].strip()
    ):
        raise ValueError("supervisor cleanup confirmation does not match this run")


def stop_lost_runner(path: Path, confirmation_path: Path | None) -> dict[str, Any]:
    root = Path(read_json(path)["session_root"])
    with (root / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = read_json(path)
        state = read_json(root / "session.json")
        if state["latest_run"] != str(path.parent):
            raise ValueError("stop recovery requires the latest session run")
        if live(result["runner"]):
            raise ValueError("runner is still alive; use ordinary stop")
        if result["result"] != "running" and not result.get("runner_lost"):
            raise ValueError("cleanup confirmation requires a lost runner")
        confirmation = (
            read_json(confirmation_path)
            if confirmation_path is not None
            else result.get("cleanup_confirmation")
        )
        if confirmation is not None:
            validate_cleanup_confirmation(confirmation, result)
        identity = result["process"]
        if (
            confirmation is not None
            and identity is not None
            and (live(identity) or _live_group_members(identity["pid"]))
        ):
            raise ValueError(
                "supervisor cleanup confirmation conflicts with live agent processes"
            )
        if identity is not None:
            if live(identity):
                result["terminated_pids"] = sorted(
                    set(
                        result["terminated_pids"]
                        + _terminate_group(None, identity["pid"], 1.0)
                    )
                )
            elif process_identity(identity["pid"]) is not None or _live_group_members(
                identity["pid"]
            ):
                raise ValueError(
                    "orphaned process identity cannot be verified; supervisor reconciliation required"
                )
        # Losing the subreaper also loses proof about tools in other sessions.
        # An empty CLI group alone must never release the resume gate.
        result.update(
            result="failed",
            runner_lost=True,
            turn_end_reason="cleanup_failed",
            exit_code=1,
            finished_at=result["finished_at"] or time.time(),
            error="runner was lost; descendant cleanup requires supervisor confirmation",
            cleanup_complete=confirmation is not None,
        )
        if confirmation is not None:
            result["cleanup_confirmation"] = confirmation
            result.update(
                turn_end_reason="stopped",
                exit_code=143,
                error="runner was lost; supervisor confirmed descendant cleanup",
            )
        try:
            recover_lost_turn(result, state)
            result.pop("recovery_error", None)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["recovery_error"] = str(exc)
            if result["cleanup_complete"]:
                result.update(turn_end_reason="metadata_error", exit_code=1)
        write_json(path, result)
        write_json(root / "session.json", state)
        return result


def inspect_or_stop(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir is None:
        raise ValueError("status/stop requires --run-dir")
    if args.action != "stop" and args.cleanup_confirmation is not None:
        raise ValueError("--cleanup-confirmation requires stop")
    path = args.run_dir.resolve() / "result.json"
    result = read_json(path)
    if args.action == "stop" and result["result"] == "running":
        if live(result["runner"]):
            if args.cleanup_confirmation is not None:
                raise ValueError("cleanup confirmation requires a lost runner")
            os.kill(result["runner"]["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                result = read_json(path)
                if result["result"] != "running":
                    break
                time.sleep(0.1)
            else:
                raise ValueError(
                    "runner did not acknowledge stop; inspect local processes"
                )
        else:
            result = stop_lost_runner(path, args.cleanup_confirmation)
    elif args.action == "stop" and result.get("runner_lost"):
        result = stop_lost_runner(path, args.cleanup_confirmation)
    elif args.cleanup_confirmation is not None:
        raise ValueError("cleanup confirmation requires a lost runner")
    summary = public_summary(result)
    summary.update(
        runner_alive=live(result["runner"]),
        agent_alive=live(result["process"]),
        agent_group_alive=bool(
            result["process"] and _live_group_members(result["process"]["pid"])
        ),
    )
    return summary


def main() -> int:
    os.umask(0o077)
    args = parser().parse_args()
    try:
        if args.action in {"status", "stop"}:
            print(json.dumps(inspect_or_stop(args), sort_keys=True))
            return 0
        result = run(args)
        print(json.dumps(public_summary(result), sort_keys=True))
        return int(result["exit_code"])
    except OSError:
        print(
            "agent_exec: runner input/state unavailable; inspect local paths and permissions",
            file=sys.stderr,
        )
        return 2
    except (ValueError, KeyError) as exc:
        print(f"agent_exec: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
