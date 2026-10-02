"""Set up the compiler cache before importing the unchanged run-24 checker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "run-i935-pipeline-candidate-r24-20260930/check_pipeline.py"


def check_environment(report: Path, inherited: dict[str, str]) -> dict[str, str]:
    """Torch may create this directory during import, before report preflight."""
    report = report.resolve()
    cache = report.with_name(report.name + "-compiler")
    return {**inherited, "TORCHINDUCTOR_CACHE_DIR": str(cache)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    command = [sys.executable, "-u", str(CHECKER), "--plan", str(args.plan)]
    if args.preflight_output is not None:
        command.extend(["--preflight-output", str(args.preflight_output)])
    os.execve(sys.executable, command, check_environment(Path(plan["report"]), dict(os.environ)))


if __name__ == "__main__":
    main()
