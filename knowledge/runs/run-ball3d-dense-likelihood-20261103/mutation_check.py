"""Inject a process-local missing prior; do not edit production files or running jobs."""

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

target = Path(__file__).resolve().parent
source = Path("src/tennis_scene/ball_likelihood/reconstruction.py")
before = hashlib.sha256(source.read_bytes()).hexdigest()
node = "tests/unit/tennis_scene/ball_likelihood/test_likelihood.py::test_acceleration_prior_reduces_flight_jitter_and_preserves_gap_and_turn_boundaries"
plugin = '''
from dataclasses import replace
def pytest_configure(config):
    import src.tennis_scene.ball_likelihood.reconstruction as module
    original = module.temporal_objective
    def no_prior(x, maps, fps, parameters):
        return original(x, maps, fps, replace(parameters, temporal_weight=0.))
    module.temporal_objective = no_prior
'''
records = []
with tempfile.TemporaryDirectory(prefix="dense-fault-") as directory:
    Path(directory, "dense_fault_probe.py").write_text(plugin)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": directory + os.pathsep + str(Path.cwd())}
    for name, extra in (("no_acceleration_prior", ["-p", "dense_fault_probe"]), ("restored", [])):
        started = time.perf_counter()
        result = subprocess.run([".venv/bin/python", "-m", "pytest", "-n", "0", "-q", *extra, node],
                                capture_output=True, text=True, env=env)
        (target / f"mutation-{name}.txt").write_text(result.stdout + result.stderr)
        records.append({"mutation": name, "node": node, "returncode": result.returncode, "elapsed_seconds": time.perf_counter() - started})
        if name == "no_acceleration_prior":
            if result.returncode != 1 or "AssertionError" not in result.stdout or "mean" not in result.stdout:
                raise ValueError("Mutation did not fail at the expected numerical assertion")
        elif result.returncode != 0:
            raise ValueError("Restored test failed")
after = hashlib.sha256(source.read_bytes()).hexdigest()
if before != after:
    raise ValueError("Mutation modified a production source file")
(target / "mutation.json").write_text(json.dumps({"kind": "process-local prior removal", "source_sha256_before": before,
                                                 "source_sha256_after": after, "cases": records}, indent=2) + "\n")
print(records)
