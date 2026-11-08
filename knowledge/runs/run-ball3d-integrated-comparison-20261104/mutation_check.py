"""Inject a candidate-decoder fault in memory; source files are never edited."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

NODE = "tests/unit/tasks/ball_detection/inference/test_heatmap_decoding.py::test_explicit_cells_refine_two_distinct_peaks_without_argmax_or_padding_bias"


class Mutation:
    def pytest_collection_modifyitems(self, items):
        import numpy as np

        import src.tasks.ball_detection.inference.heatmap_decoding as decoder

        original = decoder._local_offset

        def suppress_local_offset(image, x, y, config):
            _, code = original(image, x, y, config)
            return np.zeros(2), code

        decoder._local_offset = suppress_local_offset


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inject":
        raise SystemExit(pytest.main(["-n", "0", "-q", NODE], plugins=[Mutation()]))
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode, command in (
        ("injected", [sys.executable, __file__, "inject"]),
        ("restored", [sys.executable, "-m", "pytest", "-n", "0", "-q", NODE]),
    ):
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        (output / (mode + ".txt")).write_text(process.stdout)
        results[mode] = {"exit_code": process.returncode, "command": command}
    if (
        results["injected"]["exit_code"] != 1
        or "AssertionError" not in (output / "injected.txt").read_text()
        or results["restored"]["exit_code"] != 0
    ):
        raise ValueError(
            "Mutation was not detected by the intended numerical assertion"
        )
    (output / "mutation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
