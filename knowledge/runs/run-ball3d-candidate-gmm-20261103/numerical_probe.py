"""Trace unrounded fits and frozen baseline arrays under explicit CPU backends."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import scipy

import src.utils.geometry.triangulation as geometry
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.integration.tennis_scene.test_ball3d_candidate_gmm import node_for
from tests.support.ball3d_evaluation import make_inputs, make_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = make_inputs(frames=12)
    fits = []
    original = geometry.least_squares

    def trace(*values, **kwargs):
        fit = original(*values, **kwargs)
        fits.append(fit.x.copy())
        return fit

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, node, source = node_for(root, inputs, None)
        store = make_store(root/"store", inputs)
        artifacts = {
            "alignment": store.load(store.active("camera_alignment"), ArtifactCodec(CameraAlignmentOutput)),
            "calibration": store.load(store.active("court_calibration"), ArtifactCodec(CourtCalibrationOutput)),
            **{f"ball_{v.camera.camera_id}": v.points for v in inputs.views},
        }
        geometry.least_squares = trace
        try:
            ball = node.component.process(node.assembler.assemble(AssemblyContext(source), artifacts)).ball
        finally:
            geometry.least_squares = original
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_runtime()
    result = {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "environment": {key: os.environ.get(key) for key in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES")},
        "numpy_runtime": runtime.getvalue(), "fit_x_float64": np.asarray(fits).tolist(),
        "positions": ball.trajectory.positions.tolist(), "valid": ball.trajectory.valid.tolist(),
        "inliers": ball.trajectory.inliers.tolist(), "uv_px": ball.uv_px.tolist(),
        "reasons": ball.trajectory.reasons.tolist(), "reprojection_px": ball.trajectory.reprojection_px.tolist(),
    }
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
