"""Numerical contracts for the run-local, scoring-only error diagnostics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.scoring import annotation_oracle, error_samples
from tests.support.ball3d_evaluation import make_annotations, make_inputs

PATH = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-ball3d-error-decomposition-20261103/diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("ball3d_run_diagnostics", PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def test_position_regression_signed_slope_and_temporal_support():
    frames = np.arange(150)
    position = -0.4 + frames / 200
    x = diagnostics.position_regression(
        position, 2 + 11.25 * position, frames, repetitions=39
    )
    y = diagnostics.position_regression(
        position, -3 - 7.5 * position, frames, repetitions=39
    )
    assert x["slope_px_per_extent"] == pytest.approx(11.25)
    assert y["slope_px_per_extent"] == pytest.approx(-7.5)
    assert x["intercept_px"] == pytest.approx(2)
    assert y["intercept_px"] == pytest.approx(-3)
    np.testing.assert_allclose(x["coefficient_ci95"][1], [11.25, 11.25], atol=1e-10)
    assert x["block_count"] == 6
    short = diagnostics.position_regression(position[:20], position[:20], frames[:20])
    assert short["coefficient_ci95"] is None


def test_candidate_recovery_exact_threshold_invalid_slots_and_camera_axis():
    truth = np.array(
        [[[0.0, 0.0], [5.0, 4.0], [10.0, 1.0]], [[10.0, 10.0], [8.0, 3.0], [2.0, 7.0]]]
    )
    argmax = truth + np.array(
        [[[21, 0], [20, 0], [30, 0]], [[0, 21], [0, 50], [0, 30]]]
    )
    candidates = np.repeat((truth + 30)[..., None, :], 8, axis=2)
    candidates[0, 0, 1] = [20, 0]
    candidates[1, 0, 3] = truth[1, 0]
    candidates[0, 2, 0] = truth[0, 2]
    valid: NDArray[np.bool_] = np.ones((2, 3, 8), bool)
    valid[0, 2] = False
    scoring = np.array([[True, True, True], [True, False, False]])
    report = diagnostics.candidate_recovery(argmax, truth, scoring, candidates, valid)
    assert report["failed_argmax_count"] == 3
    assert report["recovered_count"] == [0, 1, 1, 2, 2, 2, 2, 2]
    assert report["recovered_rate"][-1] == pytest.approx(2 / 3)


def test_interpolation_and_image_speed_do_not_cross_gaps_or_split():
    uv = np.array(
        [[2.0, 1.0], [6.0, 4.0], [10.0, 7.0], [999.0, 888.0], [20.0, 2.0], [28.0, 8.0]]
    )
    valid = np.array([True, True, True, False, True, True])
    point, mask = diagnostics.interpolate_uv(
        uv, valid, [-0.2, 0.5, 1.0, 2.5, 4.5, 5.2], start=0, stop=3
    )
    np.testing.assert_array_equal(mask, [False, True, True, False, False, False])
    np.testing.assert_allclose(point[1], [4, 2.5])
    np.testing.assert_allclose(point[2], [6, 4])
    speed = diagnostics.annotation_speed(uv[None], valid[None], 10, start=2, stop=6)
    assert np.isnan(speed[0, 2:5]).all()
    assert speed[0, 5] == pytest.approx(100)


def test_event_union_and_smoothing_keep_matched_mask_and_reference_kind():
    inputs = make_inputs(frames=8)
    truth = np.stack([v.points.uv_px.astype(np.float64) for v in inputs.views])
    truth += np.array([[[3.0, 4.0]], [[5.0, 12.0]], [[7.0, 24.0]]])
    kinds: NDArray[np.str_] = np.full((3, 8), "observed", dtype="U22")
    kinds[1] = "estimated"
    annotations = make_annotations(
        inputs,
        truth,
        kinds,
        events=(
            {"event": "bounce", "frame_index": 1, "camera_id": "right"},
            {"event": "bounce", "frame_index": 2, "camera_id": "near"},
            {"event": "hit", "frame_index": 6, "camera_id": "far"},
        ),
    )
    masks = diagnostics.event_masks(
        annotations, inputs.frame_indices, start=0, stop=4, radius=0
    )
    assert np.flatnonzero(masks["bounce"]).tolist() == [1, 2]
    assert np.flatnonzero(masks["flight"]).tolist() == [0, 3]
    report = diagnostics.smoothing_report(inputs, annotations, split=4)
    row = report["tuning"]["near"]["estimated"]["bounce"]
    accepted = int(inputs.views[1].points.observed[:4].sum())
    assert row["after_px"]["count"] == accepted
    assert row["after_px"]["median"] == pytest.approx(13)
    assert report["tuning"]["near"]["observed"]["bounce"]["after_px"]["count"] == 0


def _clock_fixture():
    inputs = make_inputs(frames=48)
    offsets = np.array([0.0, 0.3, -0.4])
    uv = []
    for i, view in enumerate(inputs.views):
        t = np.arange(48) + offsets[i]
        world = np.column_stack((1 + 0.03 * t, 2 - 0.04 * t, np.ones(48)))
        center = view.camera.center
        fx, fy = view.camera.intrinsic[0, 0], view.camera.intrinsic[1, 1]
        depth = center[2] - 1
        uv.append(
            np.column_stack(
                (
                    160 + fx * (world[:, 0] - center[0]) / depth,
                    120 + fy * (center[1] - world[:, 1]) / depth,
                )
            )
        )
    annotations = make_annotations(
        inputs, np.stack(uv), np.full((3, 48), "observed", dtype="U22")
    )
    return inputs, annotations, offsets


def test_clock_fit_recovers_signed_subframe_offsets_without_evaluation_labels():
    inputs, annotations, expected = _clock_fixture()
    fitted, report = diagnostics.fit_offsets(inputs, annotations, split=24)
    np.testing.assert_allclose(fitted, expected, atol=1e-5)
    assert report["offset_seconds"] == pytest.approx([0, 0.03, -0.04])
    changed = annotations.uv_px.copy()
    changed[:, 24:] += np.array([[[13.0, -7.0]], [[-17.0, 5.0]], [[3.0, 19.0]]])
    other = make_annotations(inputs, changed, annotations.reference_kind)
    second, _ = diagnostics.fit_offsets(inputs, other, split=24)
    np.testing.assert_array_equal(fitted, second)


def test_timing_oracle_scores_original_heldout_clock_and_never_fills_label_gaps():
    inputs, annotations, offsets = _clock_fixture()
    kinds = annotations.reference_kind.copy()
    kinds[1, 30] = "occluded"
    annotations = make_annotations(inputs, annotations.uv_px, kinds)
    template = annotation_oracle(
        inputs,
        annotations,
        bundle_sha256="fixture",
        code_commit="fixture",
        code_fingerprint="fixture",
    )
    corrected = diagnostics.timing_oracle(
        inputs, annotations, offsets, template, split=24
    )
    for run in corrected[1:]:
        samples = error_samples(run, inputs, annotations)
        assert np.max(samples["error_px"]) < 1e-4
        camera = inputs.camera_ids.index(run.held_out_camera)
        expected_world = [
            1 + 0.03 * (10 + offsets[camera]),
            2 - 0.04 * (10 + offsets[camera]),
            1,
        ]
        np.testing.assert_allclose(run.result.positions[10], expected_world, atol=1e-6)
        assert run.result.diagnostics["scoring_only"] is True
    assert not corrected[1].result.valid[30]
    assert not corrected[1].result.valid[0]


def test_candidate_oracle_retains_acceptance_and_excludes_heldout_labels():
    inputs = make_inputs(frames=12)
    # Independent projection of (1,2,1), repeated on the complete source axis.
    base = np.array(
        [
            [211.42857142857142, 8.57142857142857],
            [122.22222222222223, 55.55555555555556],
            [160.0, 187.5],
        ]
    )
    truth = np.repeat(base[:, None], 12, axis=1)
    kinds: NDArray[np.str_] = np.full((3, 12), "observed", dtype="U22")
    annotations = make_annotations(inputs, truth, kinds)
    views = []
    for i, view in enumerate(inputs.views):
        peaks = np.repeat((truth[i] + [30, 0])[:, None], 8, axis=1)
        peaks[:, i + 1] = truth[i]  # Different correct rank for each camera.
        accepted: NDArray[np.bool_] = np.ones(12, bool)
        if i == 1:
            accepted[7] = False
        views.append(
            SimpleNamespace(
                camera=view.camera,
                points=SimpleNamespace(observed=accepted),
                detections=SimpleNamespace(
                    evidence=SimpleNamespace(
                        candidate_uv_px=peaks, candidate_valid=np.ones((12, 8), bool)
                    )
                ),
            )
        )
    diagnostic_input = SimpleNamespace(
        camera_ids=inputs.camera_ids,
        frame_indices=inputs.frame_indices,
        num_frames=12,
        views=views,
    )
    template = annotation_oracle(
        inputs,
        annotations,
        bundle_sha256="fixture",
        code_commit="fixture",
        code_fingerprint="fixture",
    )
    result = diagnostics.candidate_oracle(diagnostic_input, annotations, template, k=8)
    np.testing.assert_allclose(result[1].result.positions[5], [1, 2, 1], atol=1e-6)
    assert not result[1].result.valid[7]  # Near camera acceptance is retained.
    changed = truth.copy()
    changed[0] += [17, 11]
    labels_changed = make_annotations(inputs, changed, kinds)
    other = diagnostics.candidate_oracle(
        diagnostic_input, labels_changed, template, k=8
    )
    np.testing.assert_array_equal(result[1].result.positions, other[1].result.positions)
