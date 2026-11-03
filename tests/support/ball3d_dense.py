"""Asymmetric analytic heatmaps, independent of the dense interpolation code."""

from dataclasses import replace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy.special import expit

from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.utils.paths import PROJECT_ROOT
from tests.support.ball3d_evaluation import make_inputs


def dense_config(**overrides):
    """Take runtime values from the shipped Hydra composition, with test overrides."""
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        config = compose(config_name="pipeline")
    parameters = PipelineRuntimeConfig.from_config(config, bind_inputs=False).dense_likelihood
    return replace(parameters, **overrides)


def dense_inputs(*, xyz=None, log_domain=False, missing=None, distractor=None):
    if xyz is None:
        xyz = np.array([[0.37, 0.91, 1.43], [0.62, 0.81, 1.61]], np.float64)
    source = make_inputs(len(xyz))
    y, x = np.mgrid[:31, :41]
    views, projections = [], []
    for i, view in enumerate(source.views):
        c = view.camera
        centre = c.center
        # These fixtures have R=diag(1,-1,-1); do not use Camera.project.
        depth = centre[2] - xyz[:, 2]
        pixels = np.column_stack((c.intrinsic[0, 0] * (xyz[:, 0] - centre[0]) / depth + 160,
                                  c.intrinsic[1, 1] * (centre[1] - xyz[:, 1]) / depth + 120))
        projections.append(pixels)
        xy = pixels / [319, 239] * [40, 30]
        dx, dy = x[None] - xy[:, 0, None, None], y[None] - xy[:, 1, None, None]
        q = (dx / (1.15 + i * .1)) ** 2 + .23 * dx * dy + (dy / (.95 + i * .08)) ** 2
        maps = .65 * np.exp(-q / 2) if log_domain else expit(3.0 - q / 2)
        if i == missing:
            maps[:] = 0
        if i == distractor:
            maps = expit(6 - ((x[None] - 9) ** 2 + (y[None] - 23) ** 2) / 1.4)
            maps = np.repeat(maps, len(xyz), axis=0)
        maps = maps.astype(np.float32)
        config = BallCandidateConfig(max_candidates=4, nms_kernel=3, patch_size=3)
        candidates = decode_candidates(torch.from_numpy(maps)[None], config=config, subpixel_refine=False)
        evidence = BallHeatmapEvidence(source.source_size_wh, config, maps,
                                      (candidates.coords[0].numpy() * [319, 239]).astype(np.float32),
                                      candidates.scores[0].numpy(), candidates.valid[0].numpy(), candidates.cells[0].numpy(),
                                      candidates.patches[0].numpy(), candidates.patch_valid[0].numpy(),
                                      np.arange(len(xyz), dtype=np.int64), np.zeros(len(xyz), np.int64))
        views.append(replace(view, detections=replace(view.detections, evidence=evidence)))
    return replace(source, views=tuple(views)), xyz, np.stack(projections)
