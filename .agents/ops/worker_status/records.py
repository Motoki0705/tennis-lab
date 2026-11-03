"""Bounded runner reads. Never open transcripts, locks, prompts or stderr."""

from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lib.agent_events import parse_cli_event, parse_json
from lib.opsproc import process_identity

from .safety import Guard
from .task_view import run_settings

RECEIPTS = "event-receipts.jsonl"
END_REASONS = {
    "completed",
    "invalid_output",
    "metadata_error",
    "runtime_error",
    "usage_limit",
    "cli_error",
    "startup_failed",
    "timeout",
    "stopped",
    "descendants_reaped",
    "cleanup_failed",
}
SHELL_WRAPPERS = {
    ("/bin/bash", "-lc"),
    ("bash", "-lc"),
    ("/bin/sh", "-c"),
    ("sh", "-c"),
}
HTTP_URL = re.compile(
    r"""'https?://[^\s']*'|"https?://[^\s"\\$`]*"|https?://[^\s"'`;&|()<>]*""",
    re.IGNORECASE,
)
ATTACHED_OPTION_PATH = re.compile(
    r"""(?:^|[\s"'`;&|()<>])-[^\s"'`;&|()<>]*[A-Za-z0-9]/"""
)


INNER_SPECIAL_QUOTES = re.compile(r"""\$['"]""")
SHELL_QUOTE_REMOVAL = str.maketrans("", "", "'\"\\")


class RecordError(ValueError):
    """A classified private-record failure with safe public evidence fields."""

    def __init__(self, source: str, fields: list[str], code: str) -> None:
        super().__init__(code)
        self.source, self.fields, self.code = source, fields, code
        # Verified invocation settings of a run whose result is not written yet.
        self.settings: dict[str, Any] | None = None


def timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp without timezone")
    return parsed.astimezone(UTC).isoformat()


def epoch(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("invalid epoch")
    return datetime.fromtimestamp(value, UTC).isoformat()


def json_record(path: Path) -> tuple[dict[str, Any], str]:
    # Metadata has a separate hard bound; event streams use seek/tail below.
    source = "runner." + path.stem
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
            updated = epoch(os.fstat(stream.fileno()).st_mtime)
    except OSError as exc:
        raise RecordError(
            source,
            ["file"],
            "record_missing"
            if isinstance(exc, FileNotFoundError)
            else "record_unreadable",
        ) from exc
    if len(raw) > 1024 * 1024:
        raise RecordError(source, ["size_bytes"], "record_read_limit")
    try:
        value = parse_json(raw)
    except ValueError as exc:
        raise RecordError(source, ["json"], "corrupt_json") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise RecordError(source, ["schema_version"], "unsupported_schema")
    assert updated is not None
    return value, updated


def tail(path: Path, bound: int) -> tuple[list[tuple[int, int, bytes]], bool, str, int]:
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - bound)
        stream.seek(start)
        raw = stream.read(bound)
        updated = epoch(os.fstat(stream.fileno()).st_mtime)
    if start:
        # The first fragment may be the end of a large/private event.
        _, sep, rest = raw.partition(b"\n")
        start += len(raw) - len(rest)
        raw = rest if sep else b""
    lines = []
    offset = start
    for line in raw.splitlines(keepends=True):
        end = offset + len(line)
        if line.strip():
            lines.append((offset, end, line))
        offset = end
    assert updated is not None
    return lines, size > bound, updated, size


def literal_shell_words(value: str) -> list[str] | None:
    """Decode literal shell words; reject expansions and unsupported syntax."""
    words: list[str] = []
    word: list[str] = []
    quote = ""
    started = False
    index = 0
    while index < len(value):
        char = value[index]
        if quote == "'":
            if char == "'":
                quote = ""
            else:
                word.append(char)
        elif quote == '"':
            if char == '"':
                quote = ""
            elif char in "$`":
                return None
            elif char == "\\":
                index += 1
                if index == len(value):
                    return None
                escaped = value[index]
                if escaped not in '$`"\\\n':
                    word.append("\\")
                if escaped != "\n":
                    word.append(escaped)
            else:
                word.append(char)
        elif char in "'\"":
            quote = char
            started = True
        elif char in " \t":
            if started:
                words.append("".join(word))
                word = []
                started = False
                if len(words) > 3:
                    return None
        elif char in "\\$`*?[]~#{}();<>|&\n":
            # Unquoted escapes are deliberately unsupported as well.
            return None
        else:
            word.append(char)
            started = True
        index += 1
    if quote:
        return None
    if started:
        words.append("".join(word))
    return words


def unwrap_shell_command(value: Any) -> Any:
    """Unquote one known literal invocation, without evaluating its command."""
    if not isinstance(value, str):
        return value
    args = literal_shell_words(value)
    if args is not None and len(args) == 3 and (args[0], args[1]) in SHELL_WRAPPERS:
        return args[2]
    return value


def command_summary(value: Any, guard: Guard) -> dict[str, Any]:
    """Apply full-text guards and command-only path checks before publishing."""
    command = unwrap_shell_command(value)
    shown: dict[str, Any] = guard.text(command, first_line=True)
    if not isinstance(command, str) or shown["status"] != "visible":
        return shown
    candidates = [command]
    unsafe_quotes = False
    if command != value:
        # Only a removed wrapper establishes an inner command to inspect.
        continued = command.replace("\\\n", "")
        # Inspection is conservative, never an executable or displayed parse.
        inspected: dict[str, Any] = guard.text(
            continued.translate(SHELL_QUOTE_REMOVAL), first_line=True
        )
        if inspected["status"] == "hidden":
            return inspected
        # Protect quoted URL operators before removing quotes, then apply the
        # same URL exclusion to URLs newly joined by quote/backslash removal.
        inspected_paths = HTTP_URL.sub(" ", continued).translate(SHELL_QUOTE_REMOVAL)
        candidates.extend((continued, inspected_paths))
        unsafe_quotes = INNER_SPECIAL_QUOTES.search(continued) is not None
    if unsafe_quotes or any(
        ATTACHED_OPTION_PATH.search(HTTP_URL.sub(" ", candidate))
        for candidate in candidates
    ):
        return {"status": "hidden", "text": None, "truncated": False}
    return shown


def summarize(
    directory: Path, guard: Guard, *, agent: str, bound: int, tool_limit: int
) -> dict[str, Any]:
    lines, truncated, updated, event_size = tail(directory / "events.jsonl", bound)
    times: dict[tuple[int, int], tuple[str, str]] = {}
    receipt_status = "unavailable"
    receipt_path = directory / RECEIPTS
    if receipt_path.exists() or receipt_path.is_symlink():
        receipts, receipt_truncated, _, _ = tail(receipt_path, bound)
        truncated = truncated or receipt_truncated
        receipt_status = "observed"
        for _, _, raw in receipts:
            # An append may currently end in a partial JSON line. Its event is
            # still visible with unavailable timing; never invent a timestamp.
            try:
                receipt = parse_json(raw)
            except ValueError:
                if raw == receipts[-1][2] and not raw.endswith(b"\n"):
                    receipt_status = "partial"
                    continue
                raise
            if (
                not isinstance(receipt, dict)
                or set(receipt)
                != {
                    "schema_version",
                    "event_start",
                    "event_end",
                    "sha256",
                    "received_at",
                }
                or type(receipt["schema_version"]) is not int
                or receipt["schema_version"] != 1
            ):
                raise ValueError("invalid event receipt schema")
            start, end = receipt["event_start"], receipt["event_end"]
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or not isinstance(receipt["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
            ):
                raise ValueError("invalid event receipt offset")
            key = (start, end)
            if key in times:
                raise ValueError("duplicate event receipt")
            received = timestamp(receipt["received_at"])
            # The writer may have advanced after the raw tail snapshot. Leave
            # those later receipts for the next observation.
            if end <= event_size:
                times[key] = (receipt["sha256"], received)
    message = {**guard.text(None), "received_at": None, "time_status": "unavailable"}
    tools: dict[str, dict[str, Any]] = {}
    # Activity also includes receipts for oversized events outside the tail;
    # the event's reasoning/output body never needs to be loaded to show time.
    activity = max((when for _, when in times.values()), default=None)
    for index, (start, end, raw) in enumerate(lines):
        event = parse_cli_event(raw, agent)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("invalid CLI event")
        sha = hashlib.sha256(raw.removesuffix(b"\n")).hexdigest()
        timing = times.get((start, end))
        if timing and timing[0] != sha:
            raise ValueError("event receipt does not match raw event")
        when = timing[1] if timing else None
        if event["type"] == "thread.started":
            guard.remember(event.get("thread_id"))
        if not event["type"].startswith("item."):
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            raise ValueError("invalid CLI item")
        kind = item.get("type")
        if kind == "agent_message" and event["type"] == "item.completed":
            message = {
                **guard.text(item.get("text")),
                "received_at": when,
                "time_status": "observed" if when else "unavailable",
            }
        elif kind in {"command_execution", "mcp_tool_call", "file_change"}:
            tool: dict[str, Any] = {
                "kind": kind,
                "received_at": when,
                "time_status": "observed" if when else "unavailable",
            }
            if kind == "command_execution":
                tool["command"] = command_summary(item.get("command"), guard)
            elif kind == "mcp_tool_call":
                tool["server"] = guard.text(item.get("server"))
                tool["tool"] = guard.text(item.get("tool"))
            else:
                changes = item.get("changes")
                if not isinstance(changes, list):
                    raise ValueError("invalid file changes")
                tool["change_count"] = len(changes)
            key = item.get("id", f"line-{index}")
            if not isinstance(key, str):
                raise ValueError("invalid item identity")
            tools.pop(key, None)
            tools[key] = tool
    return {
        "summary": {
            "message": message,
            "tools": list(tools.values())[-tool_limit:],
            "tail_truncated": truncated,
        },
        "last_activity_at": activity,
        "activity_time_status": "observed" if activity else "unavailable",
        "receipt_status": receipt_status,
        "events_updated_at": updated,
    }


def inspect_run(
    directory: Path,
    guard: Guard,
    *,
    expected_contract: dict[str, Any],
    bound: int,
    tool_limit: int,
) -> dict[str, Any]:
    guard.remember(expected_contract.get("task_id"))
    guard.remember_path(expected_contract.get("cwd"))
    try:
        result, updated = json_record(directory / "result.json")
    except RecordError as exc:
        if exc.code != "record_missing":
            raise
        invocation, _ = json_record(directory / "invocation.json")
        verify_contract(invocation, expected_contract, "runner.invocation")
        # An invocation proves the binding and the selected settings, but
        # contains no runtime state.
        exc.settings = run_settings(invocation, "runner.invocation")
        raise exc
    verify_contract(result, expected_contract, "runner.result")
    settings = run_settings(result, "runner.result")
    if (
        result.get("run_dir") != str(directory)
        or not isinstance(result.get("session_root"), str)
        or not Path(result["session_root"]).is_absolute()
    ):
        raise RecordError(
            "runner.result", ["run_dir", "session_root"], "run_reference_mismatch"
        )
    guard.remember(result.get("run_id"))
    for path in (result.get("run_dir"), result.get("session_root"), str(directory)):
        guard.remember_path(path)
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("missing runtime")
    guard.remember(
        *(runtime.get(k) for k in ("session_id", "requested_session_id", "turn_id"))
    )
    session, session_updated = json_record(
        Path(result["session_root"]) / "session.json"
    )
    guard.remember(session.get("session_id"), session.get("last_turn_id"))
    if session.get("latest_run") != str(directory) or session.get(
        "session_id"
    ) != runtime.get("session_id"):
        raise RecordError(
            "runner.session", ["latest_run", "session_id"], "session_mismatch"
        )
    if runtime.get("turn_id") is not None and runtime["turn_id"] != session.get(
        "last_turn_id"
    ):
        raise RecordError("runner.session", ["last_turn_id"], "turn_mismatch")
    status = result.get("result")
    reason = result.get("turn_end_reason")
    if status not in {"running", "completed", "failed"} or reason not in END_REASONS | {
        None
    }:
        raise ValueError("invalid runner status")
    if (
        (status == "completed" and reason != "completed")
        or (status == "failed" and (reason is None or reason == "completed"))
        or (status == "running" and reason is not None)
    ):
        raise ValueError("contradictory runner status")
    identity = result.get("runner")
    identity_verified = False
    identity_status = "identity_unavailable"
    if identity is not None:
        if (
            not isinstance(identity, dict)
            or set(identity) != {"pid", "start_ticks", "boot_id"}
            or type(identity["pid"]) is not int
            or identity["pid"] < 1
            or not all(
                isinstance(identity[k], str) and identity[k]
                for k in ("start_ticks", "boot_id")
            )
        ):
            raise ValueError("invalid process identity")
        observed_identity = process_identity(identity["pid"])
        identity_verified = observed_identity == identity
        identity_status = (
            "identity_verified"
            if identity_verified
            else "runner_absent"
            if observed_identity is None
            else "identity_mismatch"
        )
    agent = result["contract"]["settings"]["agent"]
    if agent not in {"codex", "claude"}:
        raise ValueError("unsupported agent")
    summary = summarize(
        directory, guard, agent=agent, bound=bound, tool_limit=tool_limit
    )
    return {
        "settings": settings,
        "result": status,
        "turn_end_reason": reason,
        "identity_verified": identity_verified,
        "identity_status": identity_status,
        "started_at": epoch(result.get("started_at")),
        "finished_at": epoch(result.get("finished_at")),
        "files_updated_at": {
            "result": updated,
            "session": session_updated,
            "events": summary.pop("events_updated_at"),
        },
        **summary,
    }


def verify_contract(
    record: dict[str, Any], expected: dict[str, Any], source: str
) -> None:
    """Reject wrong/missing provenance before using any runtime status or text."""
    contract = record.get("contract")
    if not isinstance(contract, dict):
        raise RecordError(source, ["contract"], "contract_missing")
    for field in ("task_id", "cwd", "branch"):
        actual, required = contract.get(field), expected.get(field)
        if (
            not isinstance(actual, str)
            or not actual.strip()
            or not isinstance(required, str)
            or not required.strip()
        ):
            raise RecordError(source, ["contract." + field], "contract_missing")
        if field == "cwd":
            matches = (
                Path(actual).is_absolute()
                and Path(required).is_absolute()
                and Path(actual).resolve() == Path(required).resolve()
            )
        else:
            matches = actual == required
        if not matches:
            raise RecordError(source, ["contract." + field], "contract_mismatch")
