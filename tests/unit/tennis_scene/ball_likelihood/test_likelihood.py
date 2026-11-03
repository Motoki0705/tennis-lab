from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from src.tennis_scene.ball_likelihood import DenseLikelihoodConfig
from src.tennis_scene.ball_likelihood.likelihood import HeatmapLikelihood, objective
from src.tennis_scene.ball_likelihood.reconstruction import (
    event_boundaries,
    make_frame_maps,
    temporal_objective,
)
from src.tennis_scene.evaluation.ball3d.dense_likelihood import DenseHeatmapMethod
from tests.support.ball3d_dense import dense_inputs


def config(**kwargs):
    return DenseLikelihoodConfig(x_bounds=(-1., 2.), y_bounds=(-1., 2.), z_bounds=(.5, 2.5),
                                 coarse_spacing_m=.5, **kwargs)


@pytest.mark.parametrize("interpolation", ["bicubic_logit", "quadratic_logit", "quadratic_log"])
@pytest.mark.parametrize("initialization", ["candidates", "coarse"])
def test_subpixel_multiview_recovery_beats_bilinear_lattice_lock(interpolation, initialization):
    inputs, xyz, pixels = dense_inputs(log_domain=interpolation == "quadratic_log")
    cfg = config(interpolation=interpolation, initialization=initialization)
    result = DenseHeatmapMethod(cfg)(inputs)
    np.testing.assert_array_equal(result.valid, [True, True])
    errors = []
    for i, view in enumerate(inputs.views):
        projected, _ = view.camera.project(result.positions)
        errors.append(np.linalg.norm(projected - pixels[i], axis=-1))
    assert np.max(errors) < .3, errors  # original native cell spans ~8 source pixels

    maps = make_frame_maps(tuple(v.detections.evidence for v in inputs.views), tuple(v.camera for v in inputs.views), 0, cfg)

    def bilinear(point):
        score = 0.
        for m in maps:
            uv, _ = m.camera.project(point[None])
            xy = uv[0] * m.scale
            score -= float(map_coordinates(m.log_map, [[xy[1]], [xy[0]]], order=1)[0])
        return score

    linear = minimize(bilinear, xyz[0] + [.03, -.04, .02], method="Powell",
                      options={"xtol": 1e-7, "ftol": 1e-9})
    linear_error = max(np.linalg.norm(v.camera.project(linear.x[None])[0][0] - pixels[i, 0]) for i, v in enumerate(inputs.views))
    assert linear_error > max(e[0] for e in errors) + .4


@pytest.mark.parametrize("corruption", ["missing", "distractor"])
def test_one_corrupt_view_cannot_veto_two_consistent_views(corruption):
    inputs, xyz, _ = dense_inputs(**{corruption: 1})
    result = DenseHeatmapMethod(config(background=.15))(inputs)
    np.testing.assert_array_equal(result.valid, [True, True])
    np.testing.assert_allclose(result.positions, xyz, atol=.04)
    np.testing.assert_array_equal(result.used_views, [[True, True], [False, False], [True, True]])


def test_saturation_is_finite_missing_support_does_not_create_3d_and_outside_is_background():
    inputs, _, _ = dense_inputs(missing=1)
    view = inputs.views[0]
    heatmap = view.detections.evidence.heatmaps[0].copy()
    heatmap[10, 11] = 1
    m = HeatmapLikelihood(heatmap, view.camera, inputs.source_size_wh, config())
    assert m.saturated == 1 and np.isfinite(m.log_map).all()
    value, gradient, support = m.evaluate(np.array([[100., 100., 0.]]))
    np.testing.assert_allclose(value, np.log(.05))
    np.testing.assert_array_equal(gradient, [[0., 0., 0.]])
    assert not support.any()
    # Two views have no signal; one informative map cannot create a valid point.
    last = inputs.views[2]
    blank = replace(last.detections.evidence, heatmaps=np.zeros_like(last.detections.evidence.heatmaps),
                    candidate_uv_px=np.zeros_like(last.detections.evidence.candidate_uv_px),
                    candidate_scores=np.zeros_like(last.detections.evidence.candidate_scores),
                    candidate_valid=np.zeros_like(last.detections.evidence.candidate_valid),
                    candidate_cells=np.zeros_like(last.detections.evidence.candidate_cells),
                    patches=np.zeros_like(last.detections.evidence.patches), patch_valid=np.zeros_like(last.detections.evidence.patch_valid))
    result = DenseHeatmapMethod(config())(replace(inputs, views=(*inputs.views[:2], replace(last, detections=replace(last.detections, evidence=blank)))))
    assert not result.valid.any() and not result.positions.any()


def test_objective_gradient_in_metres_and_camera_keyed_temperature():
    inputs, xyz, _ = dense_inputs()
    cfg = config(view_temperatures=(("right", .8), ("far", 1.7)), view_backgrounds=(("near", .2),))
    maps = make_frame_maps(tuple(v.detections.evidence for v in inputs.views), tuple(v.camera for v in inputs.views), 0, cfg)
    point = xyz[:1] + [.018, -.031, .014]
    _, gradient, _ = objective(maps, point)
    numeric = []
    for delta in np.eye(3) * 1e-5:
        numeric.append((objective(maps, point + delta)[0][0] - objective(maps, point - delta)[0][0]) / 2e-5)
    np.testing.assert_allclose(gradient[0], numeric, atol=1e-5)
    reverse = inputs.select_cameras(("far", "right"))
    a = DenseHeatmapMethod(cfg)(inputs.select_cameras(("right", "far")))
    b = DenseHeatmapMethod(cfg)(reverse)
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-6)


def test_acceleration_prior_reduces_flight_jitter_and_preserves_gap_and_turn_boundaries():
    time = np.arange(9)
    true = np.column_stack((.3 + time * .04, .2 - time * .025, 1.3 + time * .015))
    noisy = true + np.column_stack(((-1.) ** time * .035, np.zeros(9), np.zeros(9)))
    inputs, _, _ = dense_inputs(xyz=noisy)
    raw = DenseHeatmapMethod(config())(inputs)
    cfg = config(temporal_weight=2., acceleration_scale_mps2=5., event_acceleration_mps2=1000., max_iterations=150)
    smooth = DenseHeatmapMethod(cfg)(inputs)
    assert smooth.valid.all()
    assert np.mean((smooth.positions - true) ** 2) < np.mean((raw.positions - true) ** 2) * .7
    turn = np.column_stack((np.arange(7) * .1, np.zeros(7), [1., 1.2, 1.4, 1.6, 1.4, 1.2, 1.]))
    mask: NDArray[np.bool_] = np.ones(7, bool)
    np.testing.assert_array_equal(event_boundaries(turn, mask, 10., 30.), [False, False, False, True, False, False, False])
    mask[3] = False
    assert not event_boundaries(turn, mask, 10., 30.).any()
    # The prior gradient must act on the frame axis and include FPS^2.
    maps = [make_frame_maps(tuple(v.detections.evidence for v in inputs.views), tuple(v.camera for v in inputs.views), i, cfg) for i in range(3)]
    x = noisy[:3].ravel()
    _, grad = temporal_objective(x, maps, inputs.fps, cfg)
    numeric = (temporal_objective(x + np.eye(9)[3] * 1e-5, maps, inputs.fps, cfg)[0] - temporal_objective(x - np.eye(9)[3] * 1e-5, maps, inputs.fps, cfg)[0]) / 2e-5
    assert grad[3] == pytest.approx(numeric, abs=1e-5)


def test_temporal_optimization_does_not_smooth_across_a_detected_impact_or_fill_a_gap():
    t = np.arange(9)
    xyz = np.column_stack((.4 + .03 * t, 1.1 + .01 * t, 1.2 + .12 * np.minimum(t, 8 - t)))
    inputs, _, _ = dense_inputs(xyz=xyz)
    cfg = config(temporal_weight=5., acceleration_scale_mps2=5., event_acceleration_mps2=15., max_iterations=150)
    result = DenseHeatmapMethod(cfg)(inputs)
    assert result.valid.all()
    assert result.diagnostics["event_boundaries"][4]
    np.testing.assert_allclose(result.positions, xyz, atol=.008)
    views = []
    for view in inputs.views:
        evidence = view.detections.evidence
        changed = {}
        for field in ("heatmaps", "candidate_uv_px", "candidate_scores", "candidate_valid", "candidate_cells", "patches", "patch_valid"):
            value = getattr(evidence, field).copy()
            value[4] = 0
            changed[field] = value
        views.append(replace(view, detections=replace(view.detections, evidence=replace(evidence, **changed))))
    gap = DenseHeatmapMethod(cfg)(replace(inputs, views=tuple(views)))
    np.testing.assert_array_equal(gap.valid, [True, True, True, True, False, True, True, True, True])
    np.testing.assert_array_equal(gap.positions[4], [0, 0, 0])
    np.testing.assert_allclose(gap.positions[gap.valid], xyz[gap.valid], atol=.008)


@pytest.mark.parametrize("kw", [{"temperature": 0}, {"background": 1}, {"temporal_weight": -1},
                               {"multistart": True}, {"interpolation": "bilinear"}, {"initialization": "fallback"},
                               {"coarse_spacing_m": .001}, {"view_temperatures": (("far", -1),)}])
def test_invalid_dense_config_is_rejected(kw):
    with pytest.raises(ValueError):
        DenseLikelihoodConfig(**kw)
