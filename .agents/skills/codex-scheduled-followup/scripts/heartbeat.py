"""Create/pause local Codex heartbeats and verify app registration.

Python 3.11+, standard library only. Native database files are copied, never
opened by SQLite in place. No cron job or background polling process is created.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def config_path(home: Path, task_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", task_id):
        raise ValueError(
            "id must contain lowercase letters/digits separated by hyphens"
        )
    folder = home / "automations" / task_id
    if folder.is_symlink():
        raise ValueError("Refusing a symlinked task directory")
    path = folder / "automation.toml"
    if path.is_symlink():
        raise ValueError("Refusing a symlinked automation.toml")
    return path


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = tomllib.loads(raw.decode("utf-8"))
    if value.get("version") != 1 or value.get("kind") != "heartbeat":
        raise ValueError("Only version 1 heartbeat files are supported")
    if value.get("id") != path.parent.name:
        raise ValueError("Task ID must match its directory")
    for key in ("name", "prompt", "rrule", "target_thread_id"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Missing/non-string {key}")
    if value.get("status") not in ("ACTIVE", "PAUSED"):
        raise ValueError("Only ACTIVE/PAUSED tasks are supported")
    for key in ("created_at", "updated_at"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"{key} must be a Unix timestamp in milliseconds")
    return value, raw


def atomic_write(path: Path, content: bytes, *, expected: bytes | None) -> None:
    """Publish a new file without clobbering it, or replace an unchanged file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".heartbeat-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if expected is None:
            # Atomic no-clobber publication; fail explicitly if unsupported.
            os.link(temporary, path)
        else:
            if path.read_bytes() != expected:
                raise RuntimeError(f"Configuration changed concurrently: {path}")
            os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def schedule_rrule(
    *,
    interval_hours: int | None = None,
    interval_minutes: int | None = None,
    rrule: str | None = None,
) -> str:
    """Honor one explicit schedule; only an omitted schedule defaults to hourly.

    Custom RFC 5545 text is passed to the app without interpreting its calendar
    or timezone. The app must accept it before verification can succeed.
    """
    if (
        sum(value is not None for value in (interval_hours, interval_minutes, rrule))
        > 1
    ):
        raise ValueError(
            "Specify only one of interval-hours, interval-minutes or rrule"
        )
    if rrule is not None:
        if not rrule.strip():
            raise ValueError("An explicitly supplied rrule must not be empty")
        return rrule
    for value, frequency, option in (
        (interval_hours, "HOURLY", "interval-hours"),
        (interval_minutes, "MINUTELY", "interval-minutes"),
    ):
        if value is not None:
            if type(value) is not int or value < 1:
                raise ValueError(f"{option} must be a positive integer")
            return f"FREQ={frequency};INTERVAL={value}"
    return "FREQ=HOURLY;INTERVAL=1"


def create_task(
    home: Path,
    task_id: str,
    name: str,
    prompt: str,
    thread_id: str,
    interval_hours: int | None = None,
    *,
    interval_minutes: int | None = None,
    rrule: str | None = None,
) -> dict[str, Any]:
    path = config_path(home, task_id)
    if not name.strip() or not prompt.strip() or not thread_id.strip():
        raise ValueError("name, prompt and thread ID must be nonempty")
    if thread_id != thread_id.strip():
        raise ValueError("Thread ID must not contain leading/trailing whitespace")
    desired = {
        "kind": "heartbeat",
        "name": name,
        "prompt": prompt,
        "rrule": schedule_rrule(
            interval_hours=interval_hours,
            interval_minutes=interval_minutes,
            rrule=rrule,
        ),
        "target_thread_id": thread_id,
    }
    if path.exists():
        existing, _ = load_config(path)
        if any(existing[key] != value for key, value in desired.items()):
            raise FileExistsError(
                "A different task already uses this ID; inspect it before updating"
            )
        return {
            "configuration_saved": True,
            "changed": False,
            "path": str(path),
            "status": existing["status"],
            "rrule": existing["rrule"],
            "registration_verified": False,
        }
    now = time.time_ns() // 1_000_000
    config = {
        "version": 1,
        "id": task_id,
        **desired,
        "status": "ACTIVE",
        "created_at": now,
        "updated_at": now,
    }
    text = (
        "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}"
            for key, value in config.items()
        )
        + "\n"
    )
    if tomllib.loads(text) != config:
        raise ValueError("TOML round-trip failed")
    atomic_write(path, text.encode("utf-8"), expected=None)
    return {
        "configuration_saved": True,
        "changed": True,
        "path": str(path),
        "status": "ACTIVE",
        "rrule": desired["rrule"],
        "registration_verified": False,
    }


def set_status(home: Path, task_id: str, status: str) -> dict[str, Any]:
    if status not in ("ACTIVE", "PAUSED"):
        raise ValueError("Unsupported status")
    path = config_path(home, task_id)
    config, raw = load_config(path)
    if config["status"] == status:
        return {
            "configuration_saved": True,
            "changed": False,
            "status": status,
            "registration_verified": False,
        }
    updated = max(time.time_ns() // 1_000_000, config["updated_at"] + 1)
    text = raw.decode("utf-8")
    replacements = {
        r'^status[ \t]*=[ \t]*"(?:ACTIVE|PAUSED)"[ \t]*(?:#.*)?$': f'status = "{status}"',
        r"^updated_at[ \t]*=[ \t]*[0-9]+[ \t]*(?:#.*)?$": f"updated_at = {updated}",
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError("Unexpected TOML layout; use the native management tool")
    if tomllib.loads(text) != {**config, "status": status, "updated_at": updated}:
        raise ValueError("Status edit changed unrelated configuration")
    atomic_write(path, text.encode("utf-8"), expected=raw)
    return {
        "configuration_saved": True,
        "changed": True,
        "status": status,
        "registration_verified": False,
    }


def file_signature(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def native_rows(database: Path, task_id: str) -> list[dict[str, Any]]:
    sources = (database, Path(str(database) + "-wal"))
    before = [file_signature(path) for path in sources]
    if before[0] is None:
        raise FileNotFoundError(f"App database does not exist: {database}")
    with tempfile.TemporaryDirectory(prefix="codex-heartbeat-verify-") as temporary:
        destination = Path(temporary)
        for source, signature in zip(sources, before, strict=True):
            if signature is not None:
                shutil.copyfile(source, destination / source.name)
        if before != [file_signature(path) for path in sources]:
            raise RuntimeError(
                "App database changed during snapshot; verification is inconclusive"
            )
        copied = destination / database.name
        with contextlib.closing(
            sqlite3.connect(copied.as_uri() + "?mode=ro", uri=True)
        ) as db:
            db.row_factory = sqlite3.Row
            if [row[0] for row in db.execute("PRAGMA quick_check")] != ["ok"]:
                raise RuntimeError("Database snapshot did not pass quick_check")
            columns = {row[1] for row in db.execute("PRAGMA table_info(automations)")}
            required = {
                "id",
                "name",
                "prompt",
                "kind",
                "status",
                "rrule",
                "target_thread_id",
                "next_run_at",
            }
            if not required.issubset(columns):
                raise ValueError("Unsupported native automations table schema")
            where = "id = ?"
            params = [task_id]
            if "legacy_automation_id" in columns:
                where += " OR legacy_automation_id = ?"
                params.append(task_id)
            fields = sorted(
                required
                | (
                    {"next_run_nominal_at", "last_run_at", "legacy_automation_id"}
                    & columns
                )
            )
            query = f"SELECT {', '.join(fields)} FROM automations WHERE {where}"
            return [dict(row) for row in db.execute(query, params)]


def verify_task(
    home: Path, task_id: str, database: Path, timezone: str
) -> dict[str, Any]:
    config, raw = load_config(config_path(home, task_id))
    rows = native_rows(database, task_id)
    if len(rows) != 1:
        raise ValueError(f"Expected one imported task, found {len(rows)}")
    row = rows[0]
    for key in ("name", "prompt", "kind", "status", "rrule", "target_thread_id"):
        actual, desired = row[key], config[key]
        if key == "rrule":
            actual = str(actual).removeprefix("RRULE:")
            desired = desired.removeprefix("RRULE:")
        if actual != desired:
            raise ValueError(f"Native task differs from local configuration: {key}")
    next_run = row["next_run_at"]
    if config["status"] == "ACTIVE" and (type(next_run) is not int or next_run <= 0):
        raise ValueError("Active task has no scheduled next run")
    if config["status"] == "PAUSED" and next_run is not None:
        raise ValueError("App has not cleared the paused task's next run")
    result = {key: value for key, value in row.items() if key not in ("prompt", "name")}
    result.update(
        {
            "configuration_id": task_id,
            "registration_verified": True,
            "execution_verified": False,
            "database_snapshot_quick_check": "ok",
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "checked_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    zone = ZoneInfo(timezone)
    for key in ("next_run_at", "next_run_nominal_at"):
        if result.get(key) is not None:
            result[key + "_iso"] = datetime.fromtimestamp(
                result[key] / 1000, zone
            ).isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "set-status", "verify"):
        command = commands.add_parser(action)
        command.add_argument(
            "--codex-home",
            type=Path,
            default=Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex"),
        )
        command.add_argument("--id", required=True)
        if action == "create":
            command.add_argument("--name", required=True)
            command.add_argument("--prompt-file", type=Path, required=True)
            command.add_argument(
                "--thread-id", default=os.environ.get("CODEX_THREAD_ID", "")
            )
            cadence = command.add_mutually_exclusive_group()
            cadence.add_argument(
                "--interval-hours",
                type=int,
                help="User-requested hours; default 1 only when no schedule is supplied",
            )
            cadence.add_argument(
                "--interval-minutes",
                type=int,
                help="User-requested interval in minutes",
            )
            cadence.add_argument(
                "--rrule",
                help="User-requested RFC 5545 recurrence; native app validation is required",
            )
        elif action == "set-status":
            command.add_argument(
                "--status", choices=("ACTIVE", "PAUSED"), required=True
            )
        else:
            command.add_argument("--database", type=Path)
            command.add_argument("--timezone", default="UTC")
            command.add_argument("--wait-seconds", type=float, default=0.0)
            command.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        home = args.codex_home.expanduser().resolve()
        if args.action == "create":
            result = create_task(
                home,
                args.id,
                args.name,
                args.prompt_file.read_text(encoding="utf-8"),
                args.thread_id,
                args.interval_hours,
                interval_minutes=args.interval_minutes,
                rrule=args.rrule,
            )
        elif args.action == "set-status":
            result = set_status(home, args.id, args.status)
        else:
            if not 0 <= args.wait_seconds <= 30:
                raise ValueError("wait-seconds must be between 0 and 30")
            database = args.database or home / "sqlite/codex-dev.db"
            database = database.expanduser().resolve()
            if args.receipt:
                protected = {
                    database,
                    Path(str(database) + "-wal"),
                    Path(str(database) + "-shm"),
                    config_path(home, args.id).resolve(),
                }
                if args.receipt.resolve() in protected:
                    raise ValueError("Receipt must not overwrite scheduler inputs")
            deadline = time.monotonic() + args.wait_seconds
            while True:
                try:
                    result = verify_task(home, args.id, database, args.timezone)
                    break
                except (OSError, ValueError, RuntimeError, sqlite3.Error):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            if args.receipt:
                expected = args.receipt.read_bytes() if args.receipt.exists() else None
                atomic_write(
                    args.receipt,
                    (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode(),
                    expected=expected,
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError) as error:
        print(
            json.dumps(
                {"registration_verified": False, "error": str(error)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
