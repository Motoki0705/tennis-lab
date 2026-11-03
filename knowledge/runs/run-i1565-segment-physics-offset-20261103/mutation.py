"""Temporarily reverse the clock sign; require a numerical recovery assertion."""

import json
import os
import subprocess
from pathlib import Path

DIRECTORY = Path(__file__).parent
TARGET = Path("src/tennis_scene/ball_trajectory/fit.py")
NODE = "tests/unit/tennis_scene/ball_trajectory/test_segment_physics.py::test_joint_fit_recovers_signed_offsets_drag_bounce_and_camera_reordering"

if __name__ == "__main__":
    original = TARGET.read_bytes()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    command = [".venv/bin/python", "-m", "pytest", "-n", "0", "--tb=short", "-q", NODE]
    try:
        changed = original.replace(
            b"base_times + offsets(parameters)[vv]",
            b"base_times - offsets(parameters)[vv]",
        )
        if changed == original:
            raise ValueError("Mutation target not found")
        TARGET.write_bytes(changed)
        failed = subprocess.run(command, env=env, capture_output=True, text=True)
        if failed.returncode != 1 or "Mismatched elements" not in failed.stdout:
            raise ValueError(
                f"Expected a numerical clock-recovery failure: {failed.stdout}"
            )
    finally:
        TARGET.write_bytes(original)
    restored = subprocess.run(command, env=env, capture_output=True, text=True)
    (DIRECTORY / "mutation-rejection.txt").write_text(failed.stdout + failed.stderr)
    (DIRECTORY / "mutation-restored.txt").write_text(restored.stdout + restored.stderr)
    (DIRECTORY / "mutation.json").write_text(
        json.dumps(
            {
                "node": NODE,
                "mutation": "reverse camera offset sign in image objective",
                "mutated_exit_code": failed.returncode,
                "restored_exit_code": restored.returncode,
                "bytes_restored": TARGET.read_bytes() == original,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(restored.returncode)
