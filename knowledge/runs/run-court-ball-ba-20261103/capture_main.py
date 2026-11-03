"""Capture main's original component arrays on a small asymmetric fixture."""

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    BallTriangulationInputAssembler,
)
from tests.support.court_ball_calibration import (
    calibration_pipeline_artifacts,
    main_regression_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = "src/tennis_scene/pipeline/components/triangulation.py"
    raw = subprocess.check_output(["git", "show", f"{args.main_ref}:{path}"], text=True)
    module = types.ModuleType("original_main_ball_component")
    sys.modules[module.__name__] = module
    exec(compile(raw, path, "exec"), module.__dict__)
    inputs = main_regression_inputs()
    source, artifacts = calibration_pipeline_artifacts(inputs, Path("/tmp/court-ball-main-fixture"))
    assembled = BallTriangulationInputAssembler(0.).assemble(AssemblyContext(source), artifacts)
    result = module.BallTriangulationModule(inputs.camera_ids, reprojection_px=20., min_frames=5).process(assembled).ball
    value = {
        "source_main": subprocess.check_output(["git", "rev-parse", args.main_ref], text=True).strip(),
        "component_source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "ball": {key: getattr(result, key).tolist() for key in ("uv_px", "visibility")},
        "trajectory": {key: getattr(result.trajectory, key).tolist() for key in
                       ("positions", "valid", "reasons", "inliers", "reprojection_px")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Captured {result.trajectory.valid.sum()} valid frames from {value['source_main']}")


if __name__ == "__main__":
    main()
