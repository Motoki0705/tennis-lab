"""Run one explicit, read-only API smoke; general task execution belongs to run.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.dsh import headless_config as config

TASK = """Read README.md with the read tool (first 20 lines only).
Use no other tools. Do not change files, request escalation, inspect credentials,
environment variables or absolute paths, or access the network yourself.
Reply with DSH_READ_ONLY_OK on the first line, followed by one short sentence
summarizing the README, without paths or secrets.
"""
KEY_HINT = (
    "Set DEEPSEEK_API_KEY in the launching shell or secret manager; do not put it in the repo/task/log. "
    "This smoke uses a fresh DSH_HOME and does not read existing credential files. See tools/dsh/README.md."
)
GUIDANCE = {
    "MISSING_CREDENTIAL": KEY_HINT,
    "INVALID_CREDENTIAL": "The API key is malformed. " + KEY_HINT,
    "AUTH": "Authentication failed; check/reissue the DeepSeek API key and account access. "
    + KEY_HINT,
    "PERMISSION_DENIED": "The read-only tool was denied; inspect the private tool result and file access. Do not enable danger-full-access.",
    "TOOL_ERROR": "A read-only tool failed; inspect the private result and file permissions. Do not enable danger-full-access.",
}
TOKEN_FIELDS = (
    "inputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "outputTokens",
    "totalTokens",
    "reasoningTokens",
)
# Price snapshot for this smoke only; the general runner owns shared accounting.
PRICE: dict[str, Any] = {
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "checked_date": "2026-10-19",
    "currency": "USD",
    "model": "deepseek-flash",
    "per_million_tokens": {
        "off_peak": {
            "inputTokens": 0.15,
            "cacheReadTokens": 0.003,
            "outputTokens": 0.6,
        },
        "peak": {"inputTokens": 0.30, "cacheReadTokens": 0.006, "outputTokens": 1.2},
    },
    "formula": "(inputTokens * input_miss_rate + cacheReadTokens * cache_hit_rate + outputTokens * output_rate) / 1000000",
    "time_basis": "Range across off-peak/peak; request billing time and Chinese public holidays are not resolved.",
}


class SmokeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def private_write(path: Path, value: str | bytes) -> None:
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write(value.encode("utf-8") if isinstance(value, str) else value)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def git(worktree: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        timeout=15,
    )
    if result.returncode:
        raise SmokeError(
            "GIT_ERROR",
            "Cannot inspect the worktree with git; see tools/dsh/README.md.",
        )
    return result.stdout


def git_status(worktree: Path) -> bytes:
    return git(worktree, "status", "--porcelain=v1", "--untracked-files=all", "-z")


def usage_record(events: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [
        {
            "turn": event.get("turn"),
            "step": event.get("step"),
            "usage": event.get("usage"),
        }
        for event in events
        if event.get("type") == "status" and event.get("phase") == "step_end"
    ]
    totals: dict[str, int | None] = {}
    for field in TOKEN_FIELDS:
        samples = [
            step["usage"].get(field) if isinstance(step["usage"], dict) else None
            for step in steps
        ]
        known = [n for n in samples if type(n) is int and n >= 0]
        totals[field] = sum(known) if samples and len(known) == len(samples) else None
    missing = [field for field in TOKEN_FIELDS[:4] if totals[field] is None]
    unaccounted = [
        "Auxiliary requests (session titles, compaction, etc.) may be absent from headless NDJSON.",
        "Search/Fetch and other non-token fees are not counted.",
        "This is an observed-token estimate, not the billed amount; price changes after checked_date are not tracked.",
    ]
    estimated_range = None
    if missing:
        unaccounted.append("Missing token buckets: " + ", ".join(missing))
    elif totals["cacheWriteTokens"] != 0:
        unaccounted.append("Nonzero cacheWriteTokens has no separately verified price.")
    else:
        billable = {
            field: count
            for field in ("inputTokens", "cacheReadTokens", "outputTokens")
            if (count := totals[field]) is not None
        }
        estimated_range = {
            period: sum(billable[field] * rate for field, rate in rates.items())
            / 1_000_000
            for period, rates in PRICE["per_million_tokens"].items()
        }
    return {
        "scope": "Fresh smoke session; step_end usage for this invocation only. Token buckets are disjoint.",
        "steps": steps,
        "totals": totals,
        "missing_billable_buckets": missing,
        "observed_token_estimate_range_usd": estimated_range,
        "billing_total_usd": None,
        "complete": False,
        "pricing": PRICE,
        "unaccounted": unaccounted,
    }


def parse_events(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            raise ValueError("incomplete line")
        events = [json.loads(line) for line in text.splitlines()]
        allowed = {
            "session",
            "status",
            "text",
            "thinking",
            "tool_call",
            "tool_result",
            "final",
            "error",
        }
        if not events or any(
            not isinstance(e, dict)
            or e.get("type") not in allowed
            or e.get("truncated")
            for e in events
        ):
            raise ValueError("unexpected event")
        for event in events:
            kind = event["type"]
            if kind == "status" and (
                event.get("phase")
                not in {"turn_start", "step_start", "step_end", "turn_end"}
                or type(event.get("turn")) is not int
                or event["turn"] != 1
            ):
                raise ValueError("invalid fresh-session status")
            if (
                kind == "status"
                and event["phase"] in {"step_start", "step_end"}
                and (type(event.get("step")) is not int or event["step"] < 1)
            ):
                raise ValueError("invalid step")
            if kind in {"tool_call", "tool_result"} and not isinstance(
                event.get("callId"), str
            ):
                raise ValueError("invalid callId")
            if kind == "tool_call" and (
                not isinstance(event.get("tool"), str)
                or not isinstance(event.get("input"), dict)
            ):
                raise ValueError("invalid tool input")
            if kind == "tool_result" and (
                not isinstance(event.get("result"), str)
                or event.get("status") not in {"completed", "error"}
            ):
                raise ValueError("invalid tool result")
            if (
                kind == "status"
                and event.get("phase") == "step_end"
                and "usage" in event
            ):
                usage = event["usage"]
                if (
                    not isinstance(usage, dict)
                    or any(
                        field not in usage for field in ("inputTokens", "outputTokens")
                    )
                    or any(
                        type(usage[field]) is not int or usage[field] < 0
                        for field in TOKEN_FIELDS
                        if field in usage
                    )
                ):
                    raise ValueError("invalid usage")
        return events
    except (UnicodeError, ValueError, TypeError) as error:
        raise SmokeError(
            "INVALID_NDJSON",
            "Invalid/incomplete smoke NDJSON; inspect the private events/stderr.",
        ) from error


def evaluate(events: list[dict[str, Any]], exit_code: int) -> tuple[str, str, str]:
    turns = [
        e for e in events if e.get("type") == "status" and e.get("phase") == "turn_end"
    ]
    reason = turns[-1].get("reason", {}) if turns else {}
    if isinstance(reason, dict) and reason.get("kind") == "error":
        error = reason.get("error", {})
        code = error.get("code") if isinstance(error, dict) else None
        safe_code = code if isinstance(code, str) and code in GUIDANCE else "API_ERROR"
        raise SmokeError(
            safe_code,
            GUIDANCE.get(
                safe_code, "dsh reported an API failure; inspect private events/stderr."
            ),
        )
    if exit_code != 0:
        raise SmokeError(
            "DSH_FAILED",
            "dsh exited nonzero; inspect private events/stderr and runtime configuration.",
        )
    sessions = [e for e in events if e.get("type") == "session"]
    finals = [e for e in events if e.get("type") == "final"]
    if (
        len(sessions) != 1
        or events[0] is not sessions[0]
        or not isinstance(sessions[0].get("sessionId"), str)
        or not sessions[0]["sessionId"]
        or len(turns) != 1
        or reason != {"kind": "completed"}
        or len(finals) != 1
        or events[-1] is not finals[0]
        or any(e.get("type") == "error" for e in events)
    ):
        raise SmokeError(
            "INCOMPLETE_RUN",
            "Expected one session, a completed turn_end, and one terminal final event.",
        )
    starts = [
        e.get("step")
        for e in events
        if e.get("type") == "status" and e.get("phase") == "step_start"
    ]
    ends = [
        e.get("step")
        for e in events
        if e.get("type") == "status" and e.get("phase") == "step_end"
    ]
    phases = [e["phase"] for e in events if e["type"] == "status"]
    if (
        not starts
        or starts != list(range(1, len(starts) + 1))
        or starts != ends
        or phases
        != [
            "turn_start",
            *[p for _ in starts for p in ("step_start", "step_end")],
            "turn_end",
        ]
    ):
        raise SmokeError("INCOMPLETE_RUN", "Expected complete, ordered smoke steps.")
    errors = [
        e
        for e in events
        if e.get("type") == "tool_result" and e.get("status") != "completed"
    ]
    if errors:
        denied = any(
            "[sandbox:" in str(e.get("result"))
            or "FS_SANDBOX_DENIED" in str(e.get("result"))
            for e in errors
        )
        code = "PERMISSION_DENIED" if denied else "TOOL_ERROR"
        raise SmokeError(code, GUIDANCE[code])
    calls = [e for e in events if e.get("type") == "tool_call"]
    results = [e for e in events if e.get("type") == "tool_result"]
    # This single smoke has no reason to use another tool, inspect secrets, or escalate.
    if not calls or any(
        e.get("tool") != "read"
        or e["input"].get("file_path") != "README.md"
        or set(e["input"]) - {"file_path", "offset", "limit"}
        or type(e["input"].get("limit")) is not int
        or not 1 <= e["input"]["limit"] <= 20
        or type(e["input"].get("offset", 1)) is not int
        or e["input"].get("offset", 1) != 1
        for e in calls
    ):
        raise SmokeError(
            "UNEXPECTED_TOOL",
            "The smoke did not restrict tool calls to reading README.md.",
        )
    if (
        len(results) != len(calls)
        or len({e["callId"] for e in calls}) != len(calls)
        or {e["callId"] for e in calls} != {e["callId"] for e in results}
    ):
        raise SmokeError(
            "INCOMPLETE_RUN", "The README read has no matching successful tool result."
        )
    final = finals[0].get("text")
    if (
        not isinstance(final, str)
        or not final.startswith("DSH_READ_ONLY_OK\n")
        or not final.split("\n", 1)[1].strip()
    ):
        raise SmokeError(
            "UNEXPECTED_FINAL",
            "The smoke final answer is empty or lacks the expected marker/README summary.",
        )
    return sessions[0]["sessionId"], "completed", final


def run_smoke(worktree: Path, records: Path | None) -> dict[str, Any]:
    config.permission_mode("read-only")
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise SmokeError("MISSING_CREDENTIAL", KEY_HINT)
    worktree = worktree.resolve()
    if not (worktree / ".git").is_file() or git(
        worktree, "rev-parse", "--show-toplevel"
    ).decode().strip() != str(worktree):
        raise SmokeError(
            "INVALID_WORKTREE", "Use the root of a dedicated linked git worktree."
        )
    before = git_status(worktree)
    if before:
        raise SmokeError(
            "DIRTY_WORKTREE",
            "Commit/stash changes before the read-only smoke so git status can detect any change.",
        )
    effective = config.check_config("read-only")
    run_id = "smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    records = (records or worktree / ".dsh-runs" / run_id).resolve()
    if records.is_relative_to(worktree):
        ignored = subprocess.run(
            ["git", "-C", str(worktree), "check-ignore", "-q", "--", str(records)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
        if ignored.returncode != 0:
            raise SmokeError(
                "UNSAFE_RECORDS",
                "Records inside the worktree must be git-ignored; use .dsh-runs/ or an external directory.",
            )
    records.mkdir(mode=0o700, parents=True, exist_ok=False)
    home = records / "home"
    home.mkdir(mode=0o700)
    private_write(records / "git-before.status", before)
    private_write(records / "task.md", TASK)
    arguments = config.headless_arguments("read-only")
    node = shutil.which("node")
    assert node is not None  # check_config has already required it.
    command = [node, str(config.RUNTIME), "exec", "--", *arguments, "--json", "-"]
    # The shared patch pins the official endpoint; caller DSH_HOME is intentionally
    # replaced to avoid existing profile/home patches, settings and credentials.
    environment = config.headless_environment("read-only", home, dict(os.environ))
    environment["TMPDIR"] = str(home)
    for name in ("NODE_OPTIONS", "NODE_PATH", "DSH_TOOLS_MODE"):
        environment.pop(name, None)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "config": effective,
        "passed": False,
        "exit_code": None,
    }
    metadata: dict[str, Any] = {
        "cwd": str(worktree),
        "command": command,
        "dsh_home": str(home),
        "started_utc": datetime.now(UTC).isoformat(),
    }
    final = ""
    events: list[dict[str, Any]] = []
    try:
        with (
            (records / "events.ndjson").open("xb") as stdout,
            (records / "stderr.log").open("xb") as stderr,
        ):
            (records / "events.ndjson").chmod(0o600)
            (records / "stderr.log").chmod(0o600)
            child = subprocess.Popen(
                command,
                cwd=worktree,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                child.communicate(TASK.encode(), timeout=180)
            except subprocess.TimeoutExpired as error:
                raise SmokeError(
                    "TIMEOUT",
                    "The fixed 180-second smoke limit expired; inspect the private logs.",
                ) from error
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                summary["exit_code"] = child.returncode
        events = parse_events(records / "events.ndjson")
        # Retain the final answer even for a failed run; only local private files
        # receive arbitrary provider/model text. Public output is an allowlist.
        if events[-1].get("type") == "final" and isinstance(
            events[-1].get("text"), str
        ):
            final = events[-1]["text"]
        metadata["sessions"] = [
            e.get("sessionId") for e in events if e.get("type") == "session"
        ]
        metadata["turn_ends"] = [
            e
            for e in events
            if e.get("type") == "status" and e.get("phase") == "turn_end"
        ]
        kinds = [
            e.get("reason", {}).get("kind")
            for e in metadata["turn_ends"]
            if isinstance(e.get("reason"), dict)
        ]
        summary["turn_end_reason"] = (
            kinds[-1]
            if kinds
            and isinstance(kinds[-1], str)
            and kinds[-1] in {"completed", "error", "aborted"}
            else None
        )
        session_id, reason, final = evaluate(events, child.returncode)
        metadata["session_id"] = session_id
        summary.update(
            turn_end_reason=reason,
            final_present=True,
            final_sha256=hashlib.sha256(final.encode()).hexdigest(),
            passed=True,
        )
    except SmokeError as error:
        summary.update(error_code=error.code, guidance=str(error))
    except (OSError, KeyboardInterrupt):
        summary.update(
            error_code="EXECUTION_FAILED",
            guidance="Smoke startup/interruption failure; inspect the private logs.",
        )
    finally:
        usage = usage_record(events)
        private_write(records / "usage.json", json_text(usage))
        summary["usage_totals"] = usage["totals"]
        summary["observed_token_estimate_range_usd"] = usage[
            "observed_token_estimate_range_usd"
        ]
        summary["cost_complete"] = False
        after = git_status(worktree)
        private_write(records / "git-after.status", after)
        private_write(records / "final.md", final)
        summary["worktree_unchanged"] = before == after
        if before != after:
            summary.update(
                passed=False,
                error_code="WORKTREE_CHANGED",
                guidance="git status changed during a read-only smoke; inspect and preserve the changes for review.",
            )
        metadata.update(ended_utc=datetime.now(UTC).isoformat(), **summary)
        private_write(records / "metadata.json", json_text(metadata))
        private_write(records / "summary.json", json_text(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path)
    arguments = parser.parse_args()
    os.umask(0o077)
    try:
        summary = run_smoke(arguments.worktree, arguments.records_dir)
    except (SmokeError, config.ConfigError) as error:
        summary = {
            "passed": False,
            "error_code": getattr(error, "code", "CONFIG_ERROR"),
            "guidance": str(error),
        }
    except (OSError, subprocess.SubprocessError):
        summary = {
            "passed": False,
            "error_code": "PREFLIGHT_FAILED",
            "guidance": "Cannot prepare worktree/records; check local paths, permissions and runtime installation.",
        }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
