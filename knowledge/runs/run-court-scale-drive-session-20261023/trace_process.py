"""Count rclone starts without recording credentials, configuration or argv."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    if args[:1] != ["--config"] or len(args) < 3:
        raise ValueError("Expected the skill's explicit configuration argument.")
    operation = args[2]
    if operation not in {"rcd", "lsjson", "copyto", "mkdir"}:
        raise ValueError("Unexpected operation in the bounded transport benchmark.")
    with Path(os.environ["COURT_SESSION_TRACE"]).open("a") as stream:
        stream.write(
            json.dumps(
                {"operation": operation, "phase": os.environ["COURT_SESSION_PHASE"]}
            )
            + "\n"
        )
    executable = os.environ["COURT_SESSION_RCLONE"]
    os.execv(executable, [executable, *args])


if __name__ == "__main__":
    main()
