"""Durable Codex queue submission and verified rollout receipts, without timers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

# Codex appends to rollouts. File identity/size detect replacement/truncation;
# a bounded prefix window detects nearby rewrites without hashing large logs.
WINDOW_BYTES = 64 * 1024
QUEUE_TIMEOUT_SECONDS = 180
ENV_KEYS = ("CODEX_HOME", "CODEX_SQLITE_HOME", "PATH")


def now() -> str:
    return datetime.now(UTC).isoformat()


def hash_range(handle: BinaryIO, digest: Any, start: int, end: int) -> None:
    handle.seek(start)
    remaining = end - start
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("Rollout truncated during inspection")
        digest.update(chunk)
        remaining -= len(chunk)


def window_digest(handle: BinaryIO, offset: int) -> str:
    digest = hashlib.sha256()
    hash_range(handle, digest, max(0, offset - WINDOW_BYTES), offset)
    return digest.hexdigest()


def verify_anchor(
    handle: BinaryIO, stat: os.stat_result, anchor: dict[str, Any]
) -> None:
    if [stat.st_dev, stat.st_ino] != anchor["identity"] or stat.st_size < anchor[
        "offset"
    ]:
        raise ValueError(
            "Rollout replaced, migrated, or truncated; delivery is blocked"
        )
    if window_digest(handle, anchor["offset"]) != anchor["window_sha256"]:
        raise ValueError("Rollout prefix changed; delivery is blocked")


def history(
    task: dict[str, Any],
    start: dict[str, Any] | None,
    *checks: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read complete JSONL records after `start` in one pass over the new bytes.

    `start` and every anchor in `checks` must still be intact. Reading begins at
    the end of the file when `start` is None.
    """
    path = Path(task["rollout"])
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        record = json.loads(handle.readline())
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
        for anchor in (start, *checks):
            if anchor is not None:
                verify_anchor(handle, stat, anchor)
        offset = start["offset"] if start else stat.st_size
        handle.seek(offset)
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
        end = offset + len(complete)
        snapshot = {
            "identity": [stat.st_dev, stat.st_ino],
            "offset": end,
            "window_sha256": window_digest(handle, end),
            "line_complete": os.pread(handle.fileno(), 1, stat.st_size - 1) == b"\n",
        }
    return records, snapshot


def reconcile(
    task: dict[str, Any], persist: Callable[[], None]
) -> dict[str, Any] | None:
    delivery = task["deliveries"][-1] if task["deliveries"] else None
    if task.get("blocked_reason"):
        return delivery
    try:
        observed = task["observed_boundary"]
        records, snapshot = history(
            task, delivery["boundary"] if delivery else observed, observed
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
    persist()
    return delivery


def acknowledge(
    delivery: dict[str, Any],
    thread_id: str,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> None:
    def decoded(value: str | bytes | None) -> str:
        return (
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else value or ""
        )

    output = decoded(stdout)
    delivery.update(queue_stdout=output[-4000:], queue_stderr=decoded(stderr)[-2000:])
    receipts = re.findall(
        r"^Queued message (\S+) for thread (\S+)\.$", output, re.MULTILINE
    )
    # Queue acceptance remains evidence even when CLI teardown subsequently times out.
    if len(receipts) == 1 and receipts[0][1] == thread_id:
        delivery.update(phase="queued", queue_id=receipts[0][0])


def enqueue(
    task: dict[str, Any],
    marker: str,
    *,
    persist: Callable[[], None],
    command: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    """Caller holds its lock and supplies a verified, complete observed boundary.

    Save uncertainty before invoking the CLI. An acknowledgement remains evidence
    even if teardown fails. A caller must never resend an unresolved marker.
    """
    boundary = task["observed_boundary"]
    if not boundary["line_complete"]:
        raise ValueError("Rollout has a partial record; no message was sent")
    delivery = {
        "marker": marker,
        "phase": "uncertain",
        "prepared_at": now(),
        "boundary": boundary,
    }
    task["deliveries"].append(delivery)
    persist()
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
        delivery["returncode"] = queued.returncode
        acknowledge(delivery, task["thread_id"], queued.stdout, queued.stderr)
        if delivery["phase"] == "uncertain":
            delivery["error"] = (
                "Queue acceptance is uncertain; inspect logs or use explicit recover"
            )
    except subprocess.TimeoutExpired as error:
        delivery.update(timeout_seconds=error.timeout, queue_process_timed_out=True)
        acknowledge(delivery, task["thread_id"], error.stdout, error.stderr)
        if delivery["phase"] == "uncertain":
            delivery["error"] = (
                "Queue acceptance is uncertain: TimeoutExpired; explicit recovery required"
            )
    except OSError as error:
        delivery["error"] = (
            f"Queue acceptance is uncertain: {type(error).__name__}: {error}"
        )
    persist()
    return delivery
