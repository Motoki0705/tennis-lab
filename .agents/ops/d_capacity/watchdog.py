"""Publish physical D: capacity for the isolated queue and toast on transitions."""

from __future__ import annotations

import argparse
import fcntl
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
    read_status,
    sample,
)
from lib.windows_toast import ToastError, send_windows_toast

INTERVAL_SECONDS = 30


def publish(path: Path, status: CapacityStatus) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            name = Path(stream.name)
            stream.write(status.to_json())
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
        name.replace(path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def observe(
    path: Path, notify: Callable[[str, str], None] = send_windows_toast
) -> bool:
    previous = None
    try:
        previous = read_status(path).last_measured_decision
    except FileNotFoundError:
        pass
    except CapacityError as exc:
        print(
            f"D: watchdog will replace unreadable state: {exc}",
            file=sys.stderr,
            flush=True,
        )
    status = sample(previous)
    # Persist the observation before contacting Windows so a restarted watchdog
    # cannot resend the same transition after an ambiguous notification result.
    publish(path, status)
    if status.decision == "error":
        print(
            f"D: watchdog measurement failed: {status.error}",
            file=sys.stderr,
            flush=True,
        )
        return False
    if status.decision == previous or (previous is None and status.decision == "allow"):
        return True
    assert status.available_bytes is not None and status.threshold_bytes is not None
    low = status.decision == "blocked"
    changed = "capacity low" if low else "capacity recovered"
    message = (
        f"{status.available_bytes / GIB:.2f} GiB available on Windows D: "
        f"({status.threshold_bytes / GIB:.2f} GiB threshold). "
        + ("New jobs are paused." if low else "New jobs can start.")
    )
    print(f"D: watchdog {changed}. {message}", flush=True)
    try:
        notify(f"tennis-lab D: {changed}", message)
    except ToastError as exc:
        print(f"D: watchdog toast failed: {exc}", file=sys.stderr, flush=True)
        return False
    print(f"D: watchdog submitted Windows toast ({status.decision})", flush=True)
    return True


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
