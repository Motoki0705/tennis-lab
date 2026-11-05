from dataclasses import replace

import numpy as np

from src.tennis_scene.evaluation.ball3d.baseline import PipelineBaseline
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from tests.support.ball3d_evaluation import make_inputs


def test_without_loess_keeps_acceptance_but_uses_original_detector_coordinates():
    inputs = make_inputs(frames=12)
    views = []
    for view in inputs.views:
        uv = view.detections.uv_px.copy()
        # Inlier-scale alternating image jitter. LOESS must actually do work.
        uv[view.detections.observed, 0] += np.where(
            inputs.frame_indices[view.detections.observed] % 2, 2.0, -2.0
        )
        detections = replace(view.detections, uv_px=uv)
        points = BallPointsModule(view.smoothing_config).process(
            BallPointsInput(detections, inputs.source_size_wh, inputs.fps)
        )
        views.append(replace(view, detections=detections, points=points))
    inputs = replace(inputs, views=tuple(views))
    baseline = PipelineBaseline(min_frames=1)(inputs)
    raw = PipelineBaseline(min_frames=1, without_loess=True)(inputs)
    base_diag, raw_diag = baseline.diagnostics, raw.diagnostics
    assert base_diag is not None and raw_diag is not None
    np.testing.assert_array_equal(
        base_diag["input_observed"], raw_diag["input_observed"]
    )
    for i, view in enumerate(inputs.views):
        accepted = raw_diag["input_observed"][i]
        np.testing.assert_array_equal(
            raw_diag["accepted_uv_px"][i, accepted], view.detections.uv_px[accepted]
        )
        assert np.all(raw_diag["accepted_uv_px"][i, ~accepted] == 0)
    assert (
        np.max(np.abs(base_diag["accepted_uv_px"] - raw_diag["accepted_uv_px"])) > 0.1
    )
