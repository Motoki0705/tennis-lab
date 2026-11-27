"""Inject the direct controller entry for progression tests using fake GitHub."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType


def install(gate: ModuleType) -> None:
    assert gate.__file__ is not None
    script = Path(gate.__file__).with_name("merge_gate.py")

    def start(path: Path) -> None:
        state = gate.read(path)
        with path.with_name("controller.log").open("ab") as log:
            subprocess.Popen(
                [sys.executable, str(script), "--state", str(path)],
                cwd=state["checkout"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

    gate.__dict__["start"] = start
