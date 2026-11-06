"""Temporarily remove cross curvature; require the anisotropic numeric test to fail."""

import json
import os
import subprocess
from pathlib import Path


def main():
    source = Path("src/tennis_scene/ball_gmm/heatmap.py")
    original = source.read_bytes()
    old = b"[[coef[3], coef[4]], [coef[4], coef[5]]]"
    if original.count(old) != 1:
        raise ValueError("Mutation target changed")
    command = [".venv/bin/python", "-m", "pytest", "-n", "0", "-q",
        "tests/unit/tennis_scene/ball_gmm/test_heatmap.py::test_joint_log_curvature_recovers_subpixel_and_rotated_covariance[covariance1]"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        source.write_bytes(original.replace(old, b"[[coef[3], 0.0], [0.0, coef[5]]]"))
        failed = subprocess.run(command, env=env, capture_output=True, text=True)
        if failed.returncode == 0 or "Mismatched elements" not in failed.stdout:
            raise ValueError("Cross-curvature mutation did not trigger the intended numeric assertion\n"+failed.stdout+failed.stderr)
    finally:
        source.write_bytes(original)
    restored = subprocess.run(command, env=env, capture_output=True, text=True)
    if restored.returncode or source.read_bytes() != original:
        raise ValueError("Restored numeric test failed")
    root = Path(__file__).parent
    (root/"mutation.json").write_text(json.dumps({"command": command,
        "mutation": "remove both off-diagonal precision terms", "failed_exit_code": failed.returncode,
        "failed_output": failed.stdout, "restored_exit_code": restored.returncode,
        "restored_output": restored.stdout}, indent=2)+"\n")
    print("Cross curvature mutation failed numerically; restored source passed")


if __name__ == "__main__":
    main()
