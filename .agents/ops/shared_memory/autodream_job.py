"""Run one finite AutoDream job through checkout_job.py."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    from lib.checkout_lock import require_checkout_job

    require_checkout_job()

from lib.checkout_lock import CHECKOUT_ROOT_ENV, state_root  # noqa: E402

from shared_memory.autodream import DEFAULT_THRESHOLDS, Thresholds, run  # noqa: E402
from shared_memory.autodream_plan import DEFAULT_RUNNER_TIMEOUT  # noqa: E402


class JobInterrupted(RuntimeError):
    """Keep a requested stop distinct from a retryable EINTR in subprocess I/O."""


def _interrupt(signum: int, _frame: FrameType | None) -> NoReturn:
    raise JobInterrupted(f"interrupted by {signal.Signals(signum).name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--min-days", type=int, default=DEFAULT_THRESHOLDS.days)
    parser.add_argument(
        "--min-main-commits", type=int, default=DEFAULT_THRESHOLDS.main_commits
    )
    parser.add_argument(
        "--min-memory-commits", type=int, default=DEFAULT_THRESHOLDS.memory_commits
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runner-timeout", type=float, default=DEFAULT_RUNNER_TIMEOUT)
    args = parser.parse_args()
    previous = {
        sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        result = run(
            Path(os.environ[CHECKOUT_ROOT_ENV]),
            state_dir=args.state_dir or state_root() / "shared-memory",
            remote=args.remote,
            thresholds=Thresholds(
                args.min_days, args.min_main_commits, args.min_memory_commits
            ),
            force=args.force,
            dry_run=args.dry_run,
            runner_timeout=args.runner_timeout,
        )
    except (OSError, ValueError, JobInterrupted) as exc:
        print(f"memory-autodream: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
