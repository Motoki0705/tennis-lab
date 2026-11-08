"""Capture exact pytest node/phase timings; pass normal pytest arguments after --."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest


class Receipt:
    def __init__(self):
        self.nodes = []
        self.reports = []

    def pytest_collection_modifyitems(self, items):
        self.nodes = [item.nodeid for item in items]

    def pytest_runtest_logreport(self, report):
        self.reports.append({"node": report.nodeid, "phase": report.when,
                             "outcome": report.outcome, "seconds": report.duration})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    receipt = Receipt()
    started = time.perf_counter()
    code = pytest.main(remaining, plugins=[receipt])
    args.output.write_text(json.dumps({"pytest_arguments": remaining, "exit_code": int(code),
        "wall_seconds": time.perf_counter() - started,
        "collected_count": len(receipt.nodes), "nodes": receipt.nodes, "reports": receipt.reports}, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
