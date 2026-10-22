"""Claim a saved runtime job before invoking the common runner exactly once."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parent.github import GitHub  # noqa: E402

from dispatcher.runtime import Runtime, gate  # noqa: E402
from dispatcher.schedule import Schedule  # noqa: E402
from dispatcher.state import DispatchStore  # noqa: E402
from dispatcher.supervisor import Supervisor, event, managed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    store = DispatchStore(args.store, Schedule(args.state_dir))
    with store.locked():
        dispatch = store.parents[args.key]["dispatch"]
        job = dispatch["jobs"][args.job]
        directory = Path(job["directory"])
    with gate(directory / "lifetime.lock") as fd:
        if fd is None:
            return 0
        supervisor = Supervisor(
            GitHub(args.key.rpartition("#")[0]), store, Schedule(args.state_dir)
        )
        job = supervisor.claim(args.key, args.job)
        if job is None:
            return 0
        runtime = Runtime(Path(job["runner"]))

        def stop(signum: int, frame: object) -> None:
            # Also covers TERM before the runner has written its first record.
            runtime.request_stop()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        status = runtime.execute(job, fd)
        with store.locked():
            dispatch = managed(store.parents[args.key])
            if dispatch is None:
                raise ValueError("missing managed assignment")
            dispatch["jobs"][args.job]["runner_exit"] = status
            event(dispatch, "runner_returned", job_id=args.job, exit_code=status)
            store.save()
        # A separate supervisor observes our lifetime lock released and our
        # process gone before it releases capacity. Exit 0 is not task success.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
