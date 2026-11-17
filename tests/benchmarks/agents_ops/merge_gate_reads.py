"""Measure one full automatic progression with fake GitHub and agent, real git.

The same PR count must produce the same GET count with unrelated issue families.
No live GitHub or model is used. TMPDIR must match the chosen output filesystem.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def run(checkout: Path, output: Path, repeats: int, peers: int, prs: int) -> None:
    package = ModuleType("tests")
    package.__path__ = [str(checkout / "tests")]
    sys.modules["tests"] = package
    sys.path[:0] = [str(checkout), str(checkout / ".agents/ops")]
    parent = importlib.import_module("tests.e2e.agents_ops.test_parent_session")
    api = importlib.import_module("tests.e2e.agents_ops.test_progression_cli")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(repeats):
        for count in (0, peers):
            directory = output / f"{repeat}-{count}"
            directory.mkdir()
            with pytest.MonkeyPatch.context() as patch:
                p = parent.parent_env.__wrapped__(directory, patch)
                p.env["AGENTS_TEST_GITHUB_HTTP"] = "1"
                p.env["PYTHONPATH"] = (
                    str(checkout / "tests/support/agents_ops/http_bootstrap")
                    + ":"
                    + str(checkout / ".agents/ops")
                )
                value = api.measure(p, count=prs, peers=count)
                rows.append(value)
                print(json.dumps(value), flush=True)
        assert rows[-2]["get_count"] == rows[-1]["get_count"]
    (output / "measurements.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--peers", type=int, default=30)
    parser.add_argument("--prs", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.peers < 1 or args.prs < 1:
        parser.error("repeats, peers and prs must be positive")
    run(
        args.checkout.resolve(),
        args.output.resolve(),
        args.repeats,
        args.peers,
        args.prs,
    )
