"""Make the real runner see low capacity after the caller's admission gate."""

from __future__ import annotations

import sys
from pathlib import Path

from tests.support.d_capacity import GIB, set_available
from tests.support.fake_d_capacity.sitecustomize import BYTES_FILE

REAL_RUNNER = Path(__file__).resolve().parents[3] / ".agents/ops/lib/agent_exec.sh"


def late_capacity_runner(
    root: Path, allowed_runner: Path = REAL_RUNNER
) -> tuple[Path, Path]:
    low = root / "runner-low-d-bytes"
    set_available(low, 20 * GIB - 1)
    runner = root / "capacity-runner"
    runner.write_text(
        f"""#!{sys.executable}
import os, sys
from pathlib import Path

args = sys.argv[1:]
low = Path({str(low)!r})
directory = Path(args[args.index("--run-dir") + 1])
if args[0] in {{"status", "stop"}}:
    real = (directory / "session.lock").exists()
else:
    real = low.exists()
    if real:
        os.environ[{BYTES_FILE!r}] = str(low)
runner = {str(REAL_RUNNER)!r} if real else {str(allowed_runner)!r}
os.execv(runner, [runner, *args])
""",
        encoding="utf-8",
    )
    runner.chmod(0o700)
    return runner, low
