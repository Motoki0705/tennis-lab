"""Capture collection, node identities and every pytest phase without extra fixtures."""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TARGETS = [
    "tests/unit/tasks/ball_detection/inference/test_heatmap_decoding.py",
    "tests/unit/tennis_scene/ball_association",
    "tests/unit/tennis_scene/ball_calibration",
    "tests/unit/tennis_scene/ball_gmm",
    "tests/unit/tennis_scene/ball_likelihood",
    "tests/unit/tennis_scene/ball_trajectory",
    "tests/unit/tennis_scene/evaluation/ball3d",
    "tests/unit/tennis_scene/visualization/test_ball3d_comparison.py",
    "tests/unit/tennis_scene/visualization/test_ball3d_tables.py",
    "tests/unit/tennis_scene/test_configuration.py",
    "tests/unit/tennis_scene/test_pipeline_configuration.py",
    "tests/unit/tennis_scene/pipeline/components/test_ball_detection.py",
    "tests/unit/tennis_scene/pipeline/components/test_ball_evidence.py",
    "tests/unit/tennis_scene/pipeline/components/test_ball_points.py",
    "tests/unit/tennis_scene/pipeline/test_artifact_codec.py",
    "tests/unit/utils/configuration",
    "tests/integration/tennis_scene/test_ball3d_bundle.py",
    "tests/integration/tennis_scene/test_ball3d_candidate_gmm.py",
    "tests/integration/tennis_scene/test_ball3d_dense_likelihood.py",
    "tests/integration/tennis_scene/test_ball3d_heatmap_decoding.py",
    "tests/integration/tennis_scene/test_ball3d_integrated_comparison.py",
    "tests/integration/tennis_scene/test_ball3d_media.py",
    "tests/integration/tennis_scene/test_ball3d_segment_physics.py",
    "tests/integration/tennis_scene/test_ball3d_topk.py",
    "tests/integration/tennis_scene/test_court_ball_ba.py",
    "tests/e2e/knowledge_control",
]


class Capture:
    def __init__(self):
        self.nodes = []
        self.reports = {}

    def pytest_collection_modifyitems(self, items):
        self.nodes = [i.nodeid for i in items]

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        if self.nodes and self.nodes != ids:
            raise ValueError("Workers collected different test nodes")
        self.nodes = list(ids)

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, {})[report.when] = {
            "seconds": report.duration,
            "outcome": report.outcome,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    capture = Capture()
    options = (
        ["-n", "0" if args.mode == "collect" else "2", "--collect-only"]
        if args.mode == "collect"
        else ["-n", "2", "--durations=0", "--durations-min=0"]
    )
    command = [*options, *TARGETS]
    started = time.perf_counter()
    with (
        (args.output / (args.mode + ".txt")).open("w") as stream,
        contextlib.redirect_stdout(stream),
        contextlib.redirect_stderr(stream),
    ):
        code = pytest.main(command, plugins=[capture])
    result = {
        "pytest_arguments": command,
        "wrapper_seconds": time.perf_counter() - started,
        "exit_code": int(code),
        "count": len(capture.nodes),
        "nodes": capture.nodes,
        "phases": capture.reports,
        "python": sys.version,
        "thread_limits": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    (args.output / (args.mode + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({k: result[k] for k in ("exit_code", "count", "wrapper_seconds")}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
