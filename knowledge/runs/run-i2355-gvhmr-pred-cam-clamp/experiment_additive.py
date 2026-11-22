"""Run the #2305 measurement protocol with ``run_gvhmr_additive.py``.

Only the runtime script differs from ``experiment.py``; timing, accuracy and
profiling are the #2305 / #2335 harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
for dependency in (
    HERE.parents[2],
    RUNS / "run-i2297-clipreid-tensorrt",
    RUNS / "run-i2305-gvhmr-tensorrt",
    HERE,
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import experiment  # noqa: E402

RUNTIME = RUNS / "run-i2305-gvhmr-tensorrt" / "run_gvhmr.py"
ADDITIVE_RUNTIME = HERE / "run_gvhmr_additive.py"
_runtime_command = experiment.runtime_command


def runtime_command(args, *, mode):
    command = _runtime_command(args, mode=mode)
    if Path(command[2]).resolve() != RUNTIME.resolve():
        raise RuntimeError(f"Unexpected #2305 runtime command: {command[:3]}")
    command[2] = str(ADDITIVE_RUNTIME)
    return command


if __name__ == "__main__":
    experiment.runtime_command = runtime_command
    experiment.main()
