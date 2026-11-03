import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.tennis_scene.ball_gmm.heatmap import candidate_gaussians
from tests.support.ball3d_gmm import gmm_config, make_gmm_inputs


def test_calibration_excludes_later_frames_and_unselected_camera_labels():
    fit = runpy.run_path(str(Path("knowledge/runs/run-ball3d-candidate-gmm-20261103/calibration.py")))["fit_calibration"]
    inputs, _ = make_gmm_inputs(frames=8, missing=False)
    base = gmm_config()
    candidates = tuple(candidate_gaussians(v.detections.evidence, base) for v in inputs.views[:2])
    uv = np.stack([c.means_px[:, 0] for c in candidates])
    uv[:, :, 0] += np.array([1., 2.])[:, None]
    valid = np.ones(uv.shape[:2], bool)
    first, receipt = fit(candidates, uv, valid, stop=6, base=base)
    uv[:, 6:] = 90000
    valid[:, 6:] = False
    second, _ = fit(candidates, uv, valid, stop=6, base=base)
    assert asdict(first) == asdict(second)
    assert receipt["scoreable_candidate_points"] == 12
    # Third camera observations are absent from every argument to calibration.
    assert len(candidates) == 2 and uv.shape == (2, 8, 2)
