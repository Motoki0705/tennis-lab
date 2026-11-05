"""Generate the asymmetric default regression arrays from byte-identical main code."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    reconstruct_ball,
    single_ball_observations,
)
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.utils.checksum import dual_sha256
from tests.support.ball3d_evaluation import make_inputs

MAIN = "5810fa6c0d1604cde5424640aa721b692136ecba"
FILES = [
    "src/tennis_scene/pipeline/components/ball_reconstruction.py",
    "src/tennis_scene/pipeline/components/ball_points.py",
    "src/utils/geometry/triangulation.py",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for path in FILES:
        if (
            subprocess.check_output(["git", "show", f"{MAIN}:{path}"])
            != Path(path).read_bytes()
        ):
            raise ValueError(f"The main reference implementation changed: {path}")
    inputs = make_inputs(frames=12)
    baseline = PipelineBaseline()(inputs)
    raw = ObjectObservations(
        inputs.camera_ids,
        inputs.source_size_wh,
        inputs.fps,
        np.stack([v.points.uv_px for v in inputs.views])[:, :, None, None],
        np.stack([v.points.confidence for v in inputs.views])[:, :, None, None],
        np.stack([v.points.observed for v in inputs.views])[:, :, None],
        np.zeros((3, 1), np.int64),
    )
    result = reconstruct_ball(
        single_ball_observations(raw, threshold=0),
        tuple(v.camera for v in inputs.views),
        fps=inputs.fps,
        reprojection_px=20 * np.hypot(*inputs.source_size_wh) / np.hypot(1920, 1080),
    )
    np.testing.assert_array_equal(result.trajectory.positions, baseline.positions)
    document = {
        "main_commit": MAIN,
        "source_sha256": {f: dual_sha256(Path(f)) for f in FILES},
        **{
            f: getattr(baseline, f).tolist()
            for f in ("positions", "valid", "used_views")
        },
        "accepted_uv_px": result.uv_px.tolist(),
        "rejection_codes": result.trajectory.reasons.tolist(),
        "main_component_reprojection_px": result.trajectory.reprojection_px.tolist(),
    }
    if args.output.exists():
        raise ValueError("Use a new reference output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
