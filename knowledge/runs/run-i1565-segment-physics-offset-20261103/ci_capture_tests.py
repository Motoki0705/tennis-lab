"""Preserve the original run's test evidence and capture the CI repair separately."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from capture_tests import TESTS, Records

REPAIR_TESTS = [
    *TESTS,
    "tests/unit/utils/configuration",
    "tests/unit/synthetic_data_generation/visualization/test_contracts.py",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if directory.is_relative_to(Path(__file__).parent.resolve()):
        raise ValueError("Keep full test logs outside the size-limited knowledge run")
    if not args.execute:
        directory.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path.cwd()))
    os.environ.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    os.environ["PYTHONPATH"] = (
        str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    if args.execute:
        records = Records()
        started = time.perf_counter()
        code = pytest.main(
            [
                "-n",
                "0",
                "--tb=short",
                "--durations=0",
                "--durations-min=0",
                *REPAIR_TESTS,
            ],
            plugins=[records],
        )
        (directory / "ci-test-results.json").write_text(
            json.dumps(
                {
                    "exit_code": int(code),
                    "wall_seconds": time.perf_counter() - started,
                    "phases": records.phases,
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(code)
    with (directory / "ci-test-collection.txt").open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-n",
                "0",
                "--collect-only",
                "-q",
                *REPAIR_TESTS,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (directory / "ci-test-durations.txt").open("w") as stream:
        result = subprocess.run(
            [sys.executable, __file__, "--output-dir", str(directory), "--execute"],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    report = json.loads((directory / "ci-test-results.json").read_text())
    calls = [p for p in report["phases"] if p["phase"] == "call"]
    summary = {
        "collected": len({row["nodeid"] for row in report["phases"]}),
        "passed": sum(p["outcome"] == "passed" for p in calls),
        "skipped": sum(p["outcome"] == "skipped" for p in report["phases"]),
        "exit_code": result.returncode,
        "wall_seconds": report["wall_seconds"],
    }
    (directory / "ci-test-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))
    raise SystemExit(result.returncode)
