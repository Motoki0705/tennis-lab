from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.ball_association.topk import (
    TopKConfig,
    _Hypotheses,
    associate_topk,
    frame_hypotheses,
    select_temporal_path,
)
from tests.support.ball3d_topk import candidate_evidence, make_topk_inputs


def reconstruct(inputs, config=None):
    return associate_topk(tuple(v.detections.evidence for v in inputs.views),
                          tuple(v.camera for v in inputs.views), fps=inputs.fps, config=config or TopKConfig())


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_multiview_distractor_and_sticking_recover_asymmetric_trajectory(order):
    inputs, xyz, expected_indices = make_topk_inputs()
    inputs = replace(inputs, views=tuple(inputs.views[i] for i in order))
    result = reconstruct(inputs, replace(TopKConfig(), reprojection_px=5.0))
    assert result.valid.all()
    np.testing.assert_allclose(result.positions, xyz, atol=2e-6)
    np.testing.assert_array_equal(result.selected_indices, expected_indices[list(order)])
    for i, view in enumerate(inputs.views):
        np.testing.assert_array_equal(result.selected_uv_px[i],
            view.detections.evidence.candidate_uv_px[np.arange(len(xyz)), result.selected_indices[i]])
    assert (result.selected_indices[order.index(0), 4:15] == 1).all()


def test_missing_limit_reacquires_without_fabricating_points():
    inputs, xyz, _ = make_topk_inputs(missing=((4, 6), (10, 15)))
    result = reconstruct(inputs)
    expected: NDArray[np.bool_] = np.ones(len(xyz), bool)
    expected[4:6] = expected[10:15] = False
    np.testing.assert_array_equal(result.valid, expected)
    assert not result.temporal_restart[6]
    assert result.temporal_restart[15]
    assert (result.positions[~expected] == 0).all()
    assert (result.selected_indices[:, ~expected] == -1).all()
    np.testing.assert_allclose(result.positions[expected], xyz[expected], atol=2e-6)
    # At least two genuinely selected views, even when the first view is absent.
    absent = inputs.views[0].detections.evidence
    ev = replace(absent, candidate_valid=np.zeros_like(absent.candidate_valid),
        candidate_uv_px=np.zeros_like(absent.candidate_uv_px), candidate_scores=np.zeros_like(absent.candidate_scores),
        candidate_cells=np.zeros_like(absent.candidate_cells), patches=np.zeros_like(absent.patches), patch_valid=np.zeros_like(absent.patch_valid))
    remaining = replace(inputs, views=(replace(inputs.views[0], detections=replace(inputs.views[0].detections, evidence=ev)), *inputs.views[1:]))
    pair = reconstruct(remaining)
    np.testing.assert_allclose(pair.positions[expected], xyz[expected], atol=2e-6)
    assert (pair.selected_indices[0] == -1).all()


def test_turn_cost_keeps_bounce_and_hit():
    inputs, xyz, _ = make_topk_inputs(turn=True)
    result = reconstruct(inputs, replace(TopKConfig(), max_missing_frames=0,
        acceleration_scale_mps2=1.0, turn_cost_cap=0.3))
    assert result.valid.all()
    np.testing.assert_allclose(result.positions, xyz, atol=2e-6)


def test_second_order_path_rejects_high_score_zigzag_and_uses_real_gap_time():
    rows = []
    for t in range(9):
        xyz = np.array([[t * 0.2, 0, 1], [t * 0.2, (-1) ** t * 0.7, 1]])
        rows.append(_Hypotheses(xyz, np.array([[0, 0], [1, 1]]), np.array([0.25, 0.0]), np.zeros((2, 2))))
    config = replace(TopKConfig(), max_hypotheses=2, acceleration_scale_mps2=20,
                      acceleration_weight=2, speed_weight=0, max_speed_mps=65)
    path, _ = select_temporal_path(rows, 10.0, config)
    np.testing.assert_array_equal(path, np.zeros(9, np.int64))
    # A 2m displacement across two missing frames is 6.67 m/s, not 20m/s.
    empty = _Hypotheses(np.empty((0, 3)), np.empty((0, 2), np.int64), np.empty(0), np.empty((0, 2)))
    start = _Hypotheses(rows[0].xyz[:1], rows[0].indices[:1], rows[0].cost[:1], rows[0].error[:1])
    pair = [start, empty, empty, replace(rows[3], xyz=np.array([[2., 0, 1], [30., 0, 1]]))]
    path, reset = select_temporal_path(pair, 10, replace(config, max_speed_mps=10))
    np.testing.assert_array_equal(path, [0, -1, -1, 0])
    assert not reset.any()


def test_zero_score_is_not_a_candidate_threshold():
    inputs, _, _ = make_topk_inputs()
    uv = np.stack([v.detections.evidence.candidate_uv_px[0] for v in inputs.views]).astype(float)
    hypotheses = frame_hypotheses(uv, np.zeros((3, 2)), np.ones((3, 2), bool),
                                  tuple(v.camera for v in inputs.views), TopKConfig(), 1.)
    assert len(hypotheses.cost) > 0
    assert np.isfinite(hypotheses.cost).all()


@pytest.mark.parametrize("field,value", [("top_k", 9), ("beam_width", 0), ("max_missing_frames", -1),
    ("reprojection_px", 0.), ("score_weight", float("nan")), ("top_k", True),
    ("score_weight", -1), ("acceleration_weight", True)])
def test_invalid_configuration_or_axes_fail(field, value):
    with pytest.raises(ValueError, match=field):
        replace(TopKConfig(), **{field: value})


def test_source_axes_mismatch_fails_before_association():
    inputs, _, _ = make_topk_inputs()
    evidence = tuple(v.detections.evidence for v in inputs.views)
    with pytest.raises(ValueError, match="matching"):
        associate_topk(evidence, tuple(v.camera for v in inputs.views[:2]), fps=10, config=TopKConfig())


def test_unequal_candidate_counts_do_not_drop_lower_rank_in_other_views():
    inputs, xyz, indices = make_topk_inputs()
    view = inputs.views[1]
    e = view.detections.evidence
    one = candidate_evidence(e.candidate_uv_px[:, :1], e.candidate_scores[:, :1], e.candidate_valid[:, :1])
    inputs = replace(inputs, views=(inputs.views[0], replace(view, detections=replace(view.detections, evidence=one)), inputs.views[2]))
    result = reconstruct(inputs)
    np.testing.assert_allclose(result.positions, xyz, atol=2e-6)
    np.testing.assert_array_equal(result.selected_indices, indices)
