from dataclasses import replace

import numpy as np

from src.tasks.ball_detection.inference.heatmap_decoding import HeatmapDecodingConfig
from src.tennis_scene.evaluation.ball3d.baseline import (
    PipelineBaseline,
    replay_detector,
)
from src.tennis_scene.evaluation.ball3d.heatmap_decoding import (
    HeatmapDecodingMethod,
    redecode_view,
)
from src.tennis_scene.evaluation.ball3d.runner import run_method
from tests.support.ball3d_evaluation import make_inputs


def test_legacy_redecode_matches_foundation_detector_and_runner_retains_view_order():
    inputs = make_inputs(frames=12)
    for view in inputs.views:
        actual, _ = redecode_view(view, HeatmapDecodingConfig())
        expected = replay_detector(view)
        for field in ("uv_px", "confidence", "observed", "point_kind"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    inputs = replace(inputs, views=tuple(replace(v, detector_config=replace(v.detector_config, subpixel_refine=True)) for v in inputs.views))
    method = HeatmapDecodingMethod("decoded", HeatmapDecodingConfig(method="centroid"), PipelineBaseline(min_frames=1))
    runs = run_method(method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z")
    for run in runs:
        expected_ids = tuple(c for c in inputs.camera_ids if c != run.held_out_camera)
        assert run.result.camera_ids == expected_ids
        assert run.result.diagnostics["decoding_fit_status"].shape == (len(expected_ids), 12)
    # Reordering retained views must reorder every diagnostic row, not time.
    selected = inputs.select_cameras((inputs.camera_ids[2], inputs.camera_ids[0]))
    a, b = method(selected), method(replace(selected, views=tuple(reversed(selected.views))))
    np.testing.assert_array_equal(a.diagnostics["accepted_uv_px"], b.diagnostics["accepted_uv_px"][::-1])
    # A new bundle may use the same camera IDs. The cache must not serve the old
    # view's decoded observations. Change the timeline length independently.
    other = make_inputs(frames=7)
    other = replace(other, views=tuple(replace(v, detector_config=replace(v.detector_config, subpixel_refine=True)) for v in other.views))
    fresh = method(other)
    assert fresh.frame_indices.shape == (7,)
    assert fresh.diagnostics["accepted_uv_px"].shape == (3, 7, 2)
