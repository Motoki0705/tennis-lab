"""Inject one temporal fault, require numerical failure, then restore exact bytes."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    source = Path("src/tennis_scene/ball_association/topk.py")
    destination = Path(__file__).parent
    original = source.read_bytes()
    needle = b"config.acceleration_weight * np.minimum("
    assert original.count(needle) == 1
    node = "tests/unit/tennis_scene/ball_association/test_topk.py::test_second_order_path_rejects_high_score_zigzag_and_uses_real_gap_time"
    command = [".venv/bin/python", "-m", "pytest", "-n", "0", "-q", node, "--tb=short"]
    env = {**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        source.write_bytes(original.replace(needle, b"0.0 * np.minimum("))
        failure = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=120)
        (destination / "mutation-failure.txt").write_text(failure.stdout)
        assert failure.returncode == 1 and "Mismatched elements" in failure.stdout, failure.stdout
    finally:
        source.write_bytes(original)
    assert source.read_bytes() == original
    restored = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=120)
    (destination / "mutation-restored.txt").write_text(restored.stdout)
    assert restored.returncode == 0, restored.stdout
    (destination / "mutation.json").write_text(json.dumps({
        "mutation": "zero acceleration cost in temporal transitions", "node": node, "command": command,
        "failure_returncode": failure.returncode, "numeric_assert_failure": True,
        "restored_returncode": restored.returncode,
        "restored_source_sha256": hashlib.sha256(original).hexdigest(),
        "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2) + "\n")
    print("Numeric contract rejected the injected fault; restored test passed.")


if __name__ == "__main__":
    main()
