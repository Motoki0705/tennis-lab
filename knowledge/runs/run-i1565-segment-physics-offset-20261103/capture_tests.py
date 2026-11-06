"""Record collection and every test phase without changing repository pytest policy."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).parent
TESTS = [
    "tests/unit/tennis_scene/ball_trajectory",
    "tests/integration/tennis_scene/test_ball3d_segment_physics.py",
    "tests/unit/tennis_scene/test_configuration.py",
    "tests/unit/tennis_scene/pipeline/test_orchestrator.py",
    "tests/unit/tennis_scene/pipeline/components/test_ball_points.py",
    "tests/unit/tennis_scene/evaluation/ball3d",
    "tests/unit/tennis_scene/visualization/test_ball3d_comparison.py",
    "tests/unit/tennis_scene/visualization/test_ball3d_tables.py",
    "tests/integration/tennis_scene/test_ball3d_bundle.py",
    "tests/integration/tennis_scene/test_ball3d_media.py",
]


class Records:
    def __init__(self):
        self.phases = []

    def pytest_runtest_logreport(self, report):
        self.phases.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "outcome": report.outcome,
                "duration_seconds": report.duration,
            }
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    os.environ["PYTHONPATH"] = (
        str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    if "--execute" in sys.argv:
        recorder = Records()
        start = time.perf_counter()
        code = pytest.main(
            ["-n", "0", "--tb=short", "--durations=0", "--durations-min=0", *TESTS],
            plugins=[recorder],
        )
        (DIRECTORY / "test-results.json").write_text(
            json.dumps(
                {
                    "exit_code": int(code),
                    "wall_seconds": time.perf_counter() - start,
                    "phases": recorder.phases,
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(code)
    start = time.perf_counter()
    with (DIRECTORY / "test-collection.txt").open("w") as stream:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-n", "0", "--collect-only", "-q", *TESTS],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    collection = time.perf_counter() - start
    with (DIRECTORY / "test-durations.txt").open("w") as stream:
        completed = subprocess.run(
            [sys.executable, __file__, "--execute"],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    data = json.loads((DIRECTORY / "test-results.json").read_text())
    calls = [p for p in data["phases"] if p["phase"] == "call"]
    data["collection_wall_seconds"] = collection
    data["collected_cases"] = len(calls)
    data["passed_cases"] = sum(p["outcome"] == "passed" for p in calls)
    (DIRECTORY / "test-results.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "phases"}))
    raise SystemExit(completed.returncode)
