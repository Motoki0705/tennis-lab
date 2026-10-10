"""Queue follow-ups into an existing Linux CLI and verify their rollout receipts.

Python 3.11+, standard library only. Does not resume threads or write native DBs.
Only create/pause manage systemd; tick sends at most one outstanding delivery.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENV_KEYS = ("CODEX_HOME", "CODEX_SQLITE_HOME", "PATH")
QUEUE_TIMEOUT_SECONDS = 180
TIMER_PROPERTIES = (
    "Id,LoadState,ActiveState,TimersMonotonic,NextElapseUSecRealtime,"
    "NextElapseUSecMonotonic,Triggers"
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def save(task_dir: Path, task: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=task_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(task, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, task_dir / "task.json")
        descriptor = os.open(task_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


@contextlib.contextmanager
def locked(task_dir: Path) -> Iterator[dict[str, Any]]:
    with (task_dir / "lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        if task.get("version") != 1:
            raise ValueError("Unsupported helper state version")
        yield task


def command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    kwargs.setdefault("timeout", 45)
    return subprocess.run(argv, capture_output=True, text=True, check=False, **kwargs)


def checked(argv: list[str]) -> str:
    result = command(argv)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()[:2000]}")
    return result.stdout


def history(
    task: dict[str, Any], anchor: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only append-only JSONL from the verified file and byte boundary."""
    path = Path(task["rollout"])
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        first = handle.readline()
        record = json.loads(first)
        if not isinstance(record, dict):
            raise ValueError("Unsupported rollout metadata record")
        metadata = record.get("payload", {})
        if not isinstance(metadata, dict):
            raise ValueError("Unsupported rollout metadata payload")
        if (
            record.get("type") != "session_meta"
            or metadata.get("id") != task["thread_id"]
        ):
            raise ValueError("Rollout session_meta.id does not match target thread")
        if metadata.get("cwd") != task["cwd"]:
            raise ValueError("Rollout session_meta.cwd does not match --cwd")
        offset = anchor["offset"] if anchor else stat.st_size
        if anchor and (
            [stat.st_dev, stat.st_ino] != anchor["identity"] or stat.st_size < offset
        ):
            raise ValueError(
                "Rollout replaced, migrated, or truncated; delivery is blocked"
            )
        handle.seek(0)
        prefix = hashlib.sha256()
        remaining = offset
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Rollout truncated during inspection")
            prefix.update(chunk)
            remaining -= len(chunk)
        if anchor and prefix.hexdigest() != anchor["sha256"]:
            raise ValueError("Rollout prefix changed; delivery is blocked")
        tail = handle.read(stat.st_size - offset)
        complete = tail[: tail.rfind(b"\n") + 1]
        records = [json.loads(line) for line in complete.splitlines()]
        if any(
            not isinstance(item, dict) or item.get("type") == "session_meta"
            for item in records
        ):
            raise ValueError("Unsupported rollout transition after delivery boundary")
        # A writer may still be appending the last JSON line. Anchor only the
        # complete prefix so the next inspection never starts inside that line.
        prefix.update(complete)
        if stat.st_size:
            handle.seek(stat.st_size - 1)
        snapshot = {
            "identity": [stat.st_dev, stat.st_ino],
            "offset": offset + len(complete),
            "sha256": prefix.hexdigest(),
            "line_complete": handle.read(1) == b"\n",
        }
    return records, snapshot


def reconcile(task_dir: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    delivery = task["deliveries"][-1] if task["deliveries"] else None
    if task.get("blocked_reason"):
        return delivery
    try:
        _, observed = history(task, task["observed_boundary"])
        records, snapshot = history(
            task, delivery["boundary"] if delivery else observed
        )
        if snapshot["offset"] < observed["offset"]:
            raise ValueError("Rollout truncated during receipt inspection")
        task["observed_boundary"] = snapshot
        if delivery is not None:
            current_turn = None
            matched_turn = None
            for record in records:
                event = record.get("payload", {})
                if not isinstance(event, dict):
                    raise ValueError("Unsupported rollout event payload")
                kind = event.get("type") if record.get("type") == "event_msg" else None
                user_text = ""
                if (
                    record.get("type") == "response_item"
                    and event.get("type") == "message"
                    and event.get("role") == "user"
                ):
                    user_text = "\n".join(
                        part["text"]
                        for part in event.get("content", [])
                        if isinstance(part, dict) and part.get("type") == "input_text"
                    )
                elif kind == "user_message":
                    user_text = event.get("message", "")
                if kind == "task_started":
                    current_turn = event.get("turn_id")
                elif delivery["marker"] in user_text:
                    if not isinstance(current_turn, str) or not current_turn:
                        raise ValueError(
                            "Nonce found without a preceding task_started turn_id"
                        )
                    if event.get("turn_id", current_turn) != current_turn:
                        raise ValueError("Nonce user message has a conflicting turn_id")
                    if matched_turn is not None and matched_turn != current_turn:
                        raise ValueError("Nonce was delivered in multiple turns")
                    matched_turn = current_turn
                    delivery.update(phase="started", turn_id=matched_turn)
                elif (
                    kind == "task_complete"
                    and matched_turn
                    and not isinstance(event.get("turn_id"), str)
                ):
                    raise ValueError("Unsupported task_complete without turn_id")
                elif matched_turn and event.get("turn_id") == matched_turn:
                    if kind == "task_complete":
                        delivery.update(
                            phase="completed", completed_at=record.get("timestamp")
                        )
                    elif kind in ("turn_aborted", "task_aborted"):
                        delivery.update(
                            phase="failed", error="Target turn was interrupted"
                        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        task["blocked_reason"] = str(error)
    save(task_dir, task)
    return delivery


def duration_seconds(value: str) -> float:
    units = {
        "us": 1e-6,
        "µs": 1e-6,
        "ms": 0.001,
        "s": 1,
        "min": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }
    total = 0.0
    while value.strip():
        match = re.match(r"\s*(\d+(?:\.\d+)?)\s*(min|ms|us|µs|s|h|d|w)\b", value)
        if not match:
            raise ValueError(f"Unsupported systemd timer duration: {value!r}")
        total += float(match[1]) * units[match[2]]
        value = value[match.end() :]
    return total


def timer_status(task: dict[str, Any]) -> dict[str, Any]:
    process = command(
        [
            "systemctl",
            "--user",
            "show",
            task["unit"] + ".timer",
            "--property=" + TIMER_PROPERTIES,
        ]
    )
    raw = process.stdout
    properties: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = (
                properties[key] + "\n" + value if key in properties else value
            )
    if process.returncode and properties.get("LoadState") != "not-found":
        raise RuntimeError(f"systemctl failed: {process.stderr.strip()[:2000]}")
    intervals = re.findall(
        r"(On(?:Unit)?ActiveUSec)=([^;}]+)", properties.get("TimersMonotonic", "")
    )
    interval_matches = (
        len(intervals) == 2
        and {name for name, _ in intervals} == {"OnActiveUSec", "OnUnitActiveUSec"}
        and all(
            duration_seconds(value) == task["interval_minutes"] * 60
            for _, value in intervals
        )
    )
    next_value = properties.get("NextElapseUSecMonotonic", "")
    verified = (
        properties.get("Id") == task["unit"] + ".timer"
        and properties.get("LoadState") == "loaded"
        and properties.get("ActiveState") == "active"
        and interval_matches
        and next_value not in ("", "0", "n/a", "infinity")
    )
    return {
        "registration_verified": verified,
        "actual_timer": properties,
        "timer_listing": checked(
            [
                "systemctl",
                "--user",
                "list-timers",
                task["unit"] + ".timer",
                "--all",
                "--no-pager",
            ]
        ),
    }


def result(task_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    delivery = task["deliveries"][-1] if task["deliveries"] else None
    return {
        "task_dir": str(task_dir),
        "thread_id": task["thread_id"],
        "cwd": task["cwd"],
        "interval_minutes": task["interval_minutes"],
        "unit": task["unit"],
        "status": task["status"],
        "blocked_reason": task.get("blocked_reason"),
        "delivery": delivery,
        "attention_required": bool(task.get("blocked_reason") or
            (delivery and delivery["phase"] in {"uncertain", "failed"})),
        "delivery_health": ("needs_recovery" if task.get("blocked_reason") or
            (delivery and delivery["phase"] in {"uncertain", "failed"}) else
            "waiting_for_runtime" if delivery and delivery["phase"] == "queued" else
            "in_progress" if delivery and delivery["phase"] == "started" else "ready"),
        "execution_verified": bool(
            delivery
            and delivery["phase"] == "completed"
            and not task.get("blocked_reason")
        ),
    }


def create_task(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.id):
        raise ValueError("id must contain 1–64 lowercase letters, digits or hyphens")
    if args.queue_timeout_seconds < 1:
        raise ValueError("queue-timeout-seconds must be positive")
    if args.interval_minutes < 1:
        raise ValueError("interval-minutes must be positive")
    thread_id = str(uuid.UUID(args.thread_id))
    environment = {key: os.environ.get(key, "") for key in ENV_KEYS}
    if not all(environment.values()):
        raise ValueError(
            "CODEX_HOME, CODEX_SQLITE_HOME and PATH must be explicitly set to the target CLI environment"
        )
    state_dir = args.state_dir.expanduser().resolve()
    if re.match(r"^/mnt/[a-z](?:/|$)", str(state_dir)):
        raise ValueError("state-dir must be on Linux storage, not a Windows /mnt drive")
    task_dir = state_dir / args.id
    if task_dir.exists():
        raise FileExistsError(
            f"Task already exists; inspect status, do not create another timer: {task_dir}"
        )
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex executable is unavailable")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("prompt-file must not be empty")
    cwd = str(args.cwd.expanduser().resolve(strict=True))
    task = {
        "version": 1,
        "id": args.id,
        "thread_id": thread_id,
        "cwd": cwd,
        "prompt": prompt,
        "interval_minutes": args.interval_minutes,
        "queue_timeout_seconds": args.queue_timeout_seconds,
        "environment": environment,
        "codex": str(Path(codex).resolve(strict=True)),
        "rollout": str(args.rollout.expanduser().resolve(strict=True)),
        "unit": "codex-followup-"
        + args.id
        + "-"
        + hashlib.sha256(str(task_dir).encode()).hexdigest()[:8],
        "created_at": now(),
        "status": "registering",
        "deliveries": [],
    }
    _, task["boundary"] = history(task, None)
    task["observed_boundary"] = task["boundary"]
    if not task["boundary"]["line_complete"]:
        raise ValueError(
            "Rollout ends in a partial record; retry after its write completes"
        )
    checked(["systemctl", "--user", "show", "--property=Version", "--value"])
    task_dir.mkdir(parents=True, mode=0o700)
    save(task_dir, task)
    try:
        checked(
            [
                "systemd-run",
                "--user",
                "--unit=" + task["unit"],
                "--collect",
                f"--on-active={args.interval_minutes}min",
                f"--on-unit-active={args.interval_minutes}min",
                "--timer-property=AccuracySec=1s",
                "--property=Type=oneshot",
                "--working-directory=" + cwd,
                "--expand-environment=no",
                "--",
                os.path.abspath(sys.executable),
                str(Path(__file__).resolve()),
                "tick",
                "--task-dir",
                str(task_dir),
            ]
        )
        observed = timer_status(task)
        if not observed["registration_verified"]:
            raise RuntimeError(
                "Timer registration/interval/next run did not verify; inspect status or pause"
            )
        task["status"] = "active"
        save(task_dir, task)
        return {**result(task_dir, task), **observed}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        task.update(status="registration_failed", blocked_reason=str(error))
        save(task_dir, task)
        raise


def acknowledge(delivery: dict[str, Any], thread_id: str,
                stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    def decoded(value: str | bytes | None) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
    output = decoded(stdout)
    delivery.update(queue_stdout=output[-4000:], queue_stderr=decoded(stderr)[-2000:])
    receipts = re.findall(r"^Queued message (\S+) for thread (\S+)\.$", output, re.MULTILINE)
    # Queue acceptance remains evidence even when CLI teardown subsequently times out.
    if len(receipts) == 1 and receipts[0][1] == thread_id:
        delivery.update(phase="queued", queue_id=receipts[0][0])


def recover(task_dir: Path, *, marker: str, reason: str, allow_duplicate: bool) -> dict[str, Any]:
    """Operator-authorized recovery; preserve the uncertain attempt without claiming it failed."""
    if not allow_duplicate or not reason.strip():
        raise ValueError("Recovery requires a reason and --acknowledge-possible-duplicate")
    with locked(task_dir) as task:
        delivery = reconcile(task_dir, task)
        if task.get("blocked_reason") or task["status"] != "active":
            raise ValueError("Resolve the history/registration problem before delivery recovery")
        if delivery is None or delivery["marker"] != marker:
            raise ValueError("Recovery marker must match the latest delivery")
        if delivery["phase"] not in {"uncertain", "failed"}:
            raise ValueError("Only an uncertain or interrupted delivery can be recovered; queued/started messages must wait")
        delivery["recovery"] = dict(at=now(), previous_phase=delivery["phase"], reason=reason.strip(),
                                    duplicate_risk_acknowledged=True, queue_item_deleted=False)
        delivery["phase"] = "superseded"
        save(task_dir, task)
        return {**result(task_dir, task), "recovered": True, "enqueued": False,
                "notice": "Old receipt retained. A late delivery remains possible; tick may now submit a new marker."}


def tick(task_dir: Path) -> dict[str, Any]:
    with locked(task_dir) as task:
        delivery = reconcile(task_dir, task)
        if (
            task["status"] != "active"
            or task.get("blocked_reason")
            or (delivery and delivery["phase"] not in {"completed", "superseded"})
        ):
            return {**result(task_dir, task), "enqueued": False}
        _, boundary = history(
            task, delivery["boundary"] if delivery else task["boundary"]
        )
        if not boundary["line_complete"]:
            raise ValueError("Rollout has a partial record; no message was sent")
        marker = f"[codex-heartbeat:{task['id']}:{uuid.uuid4()}]"
        delivery = {
            "marker": marker,
            "phase": "uncertain",
            "prepared_at": now(),
            "boundary": boundary,
        }
        task["deliveries"].append(delivery)
        task["observed_boundary"] = boundary
        save(task_dir, task)  # Durable uncertain receipt must precede the subprocess.
        environment = {**os.environ, **task["environment"]}
        try:
            queued = command(
                [
                    task["codex"],
                    "--disable",
                    "daemon_auto_start",
                    "queue",
                    "--thread",
                    task["thread_id"],
                    "--message",
                    marker + "\n" + task["prompt"],
                ],
                env=environment,
                cwd=task["cwd"],
                timeout=task.get("queue_timeout_seconds", QUEUE_TIMEOUT_SECONDS),
            )
            delivery.update(
                queue_stdout=queued.stdout[-4000:],
                queue_stderr=queued.stderr[-2000:],
                returncode=queued.returncode,
            )
            acknowledge(delivery, task["thread_id"], queued.stdout, queued.stderr)
            if delivery["phase"] == "uncertain":
                delivery["error"] = "Queue acceptance is uncertain; inspect logs or use explicit recover"
        except subprocess.TimeoutExpired as error:
            delivery.update(timeout_seconds=error.timeout, queue_process_timed_out=True)
            acknowledge(delivery, task["thread_id"], error.stdout, error.stderr)
            if delivery["phase"] == "uncertain":
                delivery["error"] = "Queue acceptance is uncertain: TimeoutExpired; explicit recovery required"
        except OSError as error:
            delivery["error"] = f"Queue acceptance is uncertain: {type(error).__name__}: {error}"
        save(task_dir, task)
        reconcile(task_dir, task)
        return {
            **result(task_dir, task),
            "enqueued": delivery["phase"] in ("queued", "started", "completed"),
        }


def status(task_dir: Path) -> dict[str, Any]:
    with locked(task_dir) as task:
        reconcile(task_dir, task)
        try:
            observed = timer_status(task)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            observed = {"registration_verified": False, "timer_error": str(error)}
        return {**result(task_dir, task), **observed}


def pause(task_dir: Path) -> dict[str, Any]:
    with locked(task_dir) as task:
        task["status"] = (
            "paused"  # Any already-starting helper will also see this state.
        )
        save(task_dir, task)
        checked(["systemctl", "--user", "stop", task["unit"] + ".timer"])
        observed = timer_status(task)
        timer = observed["actual_timer"]
        if timer.get("ActiveState") != "inactive" or timer.get(
            "NextElapseUSecMonotonic", ""
        ) not in ("", "0", "n/a", "infinity"):
            raise RuntimeError("Timer stop/cleared next run did not verify")
        return {
            **result(task_dir, task),
            **observed,
            "pause_verified": True,
            "notice": "Already accepted queue messages may still run; no native queue item was deleted.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("create")
    for name in ("id", "thread-id"):
        create.add_argument("--" + name, required=True)
    for name in ("prompt-file", "rollout", "cwd"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--interval-minutes", type=int, default=60)
    create.add_argument("--queue-timeout-seconds", type=int, default=QUEUE_TIMEOUT_SECONDS)
    create.add_argument(
        "--state-dir", type=Path, default=Path.home() / ".local/state/codex-followups"
    )
    for action in ("tick", "send", "status", "pause"):
        commands.add_parser(action).add_argument("--task-dir", type=Path, required=True)
    recovery = commands.add_parser("recover")
    recovery.add_argument("--task-dir", type=Path, required=True)
    recovery.add_argument("--delivery-marker", required=True)
    recovery.add_argument("--reason", required=True)
    recovery.add_argument("--acknowledge-possible-duplicate", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "create":
            output = create_task(args)
        elif args.action == "recover":
            output = recover(args.task_dir.expanduser().resolve(strict=True), marker=args.delivery_marker,
                             reason=args.reason, allow_duplicate=args.acknowledge_possible_duplicate)
        else:
            operation = {"tick": tick, "send": tick, "status": status, "pause": pause}[
                args.action
            ]
            output = operation(args.task_dir.expanduser().resolve(strict=True))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.action == "pause" and output.get("pause_verified"):
            # Retain prior delivery/registration failures without misreporting
            # the independently verified stop operation as a failure.
            return 0
        return (
            1
            if output.get("blocked_reason")
            or (output.get("delivery") or {}).get("phase") in ("uncertain", "failed")
            else 0
        )
    except BlockingIOError:
        print(json.dumps({"status": "busy", "enqueued": False}))
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {"error": str(error), "execution_verified": False}, ensure_ascii=False
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
