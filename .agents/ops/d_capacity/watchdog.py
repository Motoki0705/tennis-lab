"""Publish physical D: capacity for the isolated queue and toast on transitions."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from lib.d_capacity import (
    GIB,
    STATE_PATH,
    CapacityError,
    CapacityStatus,
    sample,
)
from lib.windows_toast import ToastError, ToastNotSent, send_windows_toast

INTERVAL_SECONDS = 30


def write_state(path: Path, text: str, *, durable: bool = False) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            name = Path(stream.name)
            stream.write(text)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
            else:
                os.fchmod(stream.fileno(), 0o644)
        name.replace(path)
        if durable:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def read_notification_state(path: Path) -> tuple[str | None, bool]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, False
    if (
        not isinstance(state, dict)
        or state.keys() != {"decision", "retry"}
        or state["decision"] not in ("allow", "blocked")
        or type(state["retry"]) is not bool
    ):
        raise ValueError(f"invalid Windows notification state: {path}")
    return state["decision"], state["retry"]


def notify_transition(
    history: Path, status: CapacityStatus, notify: Callable[[str, str], None]
) -> bool:
    previous, retry = read_notification_state(history)
    if (status.decision == previous and not retry) or (
        previous is None and status.decision == "allow"
    ):
        return True

    def save_attempt(*, retry: bool) -> None:
        write_state(
            history,
            json.dumps({"decision": status.decision, "retry": retry}) + "\n",
            durable=True,
        )

    assert status.available_bytes is not None and status.threshold_bytes is not None
    low = status.decision == "blocked"
    changed = "capacity low" if low else "capacity recovered"
    message = (
        f"{status.available_bytes / GIB:.2f} GiB available on Windows D: "
        f"({status.threshold_bytes / GIB:.2f} GiB threshold). "
        + ("New jobs are paused." if low else "New jobs can start.")
    )
    print(f"D: watchdog {changed}. {message}", flush=True)
    # A crash while submitting the toast leaves delivery uncertain.
    save_attempt(retry=False)
    try:
        notify(f"tennis-lab D: {changed}", message)
    except ToastNotSent as exc:
        print(
            f"D: watchdog toast not sent. Will retry ({exc})",
            file=sys.stderr,
            flush=True,
        )
        save_attempt(retry=True)
        return False
    except ToastError as exc:
        print(
            f"D: watchdog toast delivery uncertain. Duplicate protection applied ({exc})",
            file=sys.stderr,
            flush=True,
        )
        return False
    print(f"D: watchdog submitted Windows toast ({status.decision})", flush=True)
    return True


def observe(
    path: Path, notify: Callable[[str, str], None] = send_windows_toast
) -> bool:
    try:
        status = sample()
    except CapacityError as exc:
        path.unlink(missing_ok=True)
        print(f"D: watchdog sample unavailable: {exc}", file=sys.stderr, flush=True)
        return False
    write_state(path, status.to_json())
    if status.decision == "error":
        print(
            f"D: watchdog measurement failed: {status.error}",
            file=sys.stderr,
            flush=True,
        )
        return False
    state_home = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    history = state_home / "tennis-lab-agents/d-capacity/notification.json"
    try:
        history.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with history.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return notify_transition(history, status, notify)
    except (OSError, ValueError) as exc:
        print(
            f"D: watchdog notification state unavailable at {history}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="sample once and exit")
    parser.add_argument("--state-file", type=Path, default=STATE_PATH)
    args = parser.parse_args()
    try:
        with args.state_file.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.once:
                return 0 if observe(args.state_file) else 1
            while True:
                observe(args.state_file)
                time.sleep(INTERVAL_SECONDS)
    except OSError as exc:
        print(
            f"D: watchdog state unavailable at {args.state_file}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
