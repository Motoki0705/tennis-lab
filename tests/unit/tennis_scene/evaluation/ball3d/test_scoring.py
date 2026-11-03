from dataclasses import fields, replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.annotations import Ball2DAnnotations
from src.tennis_scene.evaluation.ball3d.contracts import Ball3DResult
from src.tennis_scene.evaluation.ball3d.runner import run_method
from src.tennis_scene.evaluation.ball3d.scoring import (
    ScoringConfig,
    annotation_oracle,
    error_samples,
    motion_samples,
    project_result,
    score_comparison,
)
from tests.support.ball3d_evaluation import make_inputs


class FixedPointMethod:
    name = "independent_fixture"
    config = {"fixture": "hand-calculated pinhole projections"}

    def __init__(self):
        self.received = []

    def __call__(self, inputs):
        self.received.append(inputs)
        valid = np.array([True, True, True, False])
        xyz = np.tile(np.array([1., 2., 1.], np.float32), (4, 1))
        xyz[~valid] = 0
        return Ball3DResult(inputs.camera_ids, inputs.frame_indices, xyz, valid,
                            np.tile(valid, (len(inputs.views), 1)))


def _labels(inputs):
    # Derived on paper from the three distinct focal lengths and camera centers:
    # right: (160+120*3/7,120-130*6/7); near: (160-170*2/9,120-145*4/9);
    # far: (160,120+180*3/8). These do not call the production projector.
    base = np.array([[211.42857142857142, 8.57142857142857],
                     [122.22222222222223, 55.55555555555556], [160., 187.5]])
    uv: NDArray[np.float64] = np.zeros((3, 4, 2), np.float64)
    states: NDArray[np.uint8] = np.zeros((3, 4), np.uint8)
    uv[:, 0] = base + [[3, 4], [5, 12], [7, 24]]  # Euclidean distances 5, 13, 25.
    uv[:2, 2] = base[:2] + [[0, 0], [3, 4]]      # 0, 5; far is occluded.
    uv[:, 3] = base + [[7, 24], [5, 12], [3, 4]]  # Result invalid: never scored.
    states[:, [0, 3]] = 1
    states[:2, 2] = 1
    states[2, 2] = 2
    return Ball2DAnnotations(inputs.camera_ids, inputs.frame_indices, inputs.source_size_wh, inputs.fps,
                             uv, states, {"kind": "synthetic_test_fixture"})


def test_lovo_heldout_pixels_missingness_and_time_splits_match_hand_calculation():
    inputs, method = make_inputs(), FixedPointMethod()
    runs = run_method(method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z")
    labels = _labels(inputs)
    report = score_comparison({method.name: runs}, inputs, labels, config=ScoringConfig(split_frame=2))
    metrics = report["methods"][method.name]
    tune, evaluate, all_frames = [metrics["lovo_reprojection_px"][key] for key in ("tuning", "evaluation", "all")]
    assert (tune["count"], evaluate["count"], all_frames["count"]) == (3, 2, 5)
    assert tune["median"] == pytest.approx(13)
    assert tune["p90"] == pytest.approx(22.6)
    assert tune["above_threshold_rate"] == pytest.approx(1 / 3)
    assert evaluate["median"] == pytest.approx(2.5)
    assert evaluate["p90"] == pytest.approx(4.5)
    assert all_frames["median"] == pytest.approx(5)
    assert all_frames["p90"] == pytest.approx(20.2)
    assert all_frames["above_threshold_rate"] == pytest.approx(1 / 5)
    assert metrics["folds"]["all_views"]["evaluation"]["scored_label_coverage"] == pytest.approx(2 / 5)
    assert metrics["folds"]["without_far"]["evaluation"]["reprojection_px"]["count"] == 0
    # Change only non-held-out labels; the right-camera fold must be unchanged.
    changed = labels.uv_px.copy()
    changed[1:, 0] = [20, 20]
    other = replace(labels, uv_px=changed)
    np.testing.assert_array_equal(error_samples(runs[1], inputs, labels)["error_px"],
                                   error_samples(runs[1], inputs, other)["error_px"])


def test_method_runner_excludes_labels_reference_and_heldout_camera():
    inputs, method = make_inputs(), FixedPointMethod()
    first = run_method(method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z")
    score_comparison({method.name: first}, inputs, _labels(inputs), config=ScoringConfig(split_frame=2))
    second = run_method(method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z")
    assert [row.camera_ids for row in method.received[:4]] == [
        ("right", "near", "far"), ("near", "far"), ("right", "far"), ("right", "near")]
    assert {field.name for field in fields(method.received[1])} == {
        "clip_id", "frame_indices", "fps", "source_size_wh", "views"}
    assert {view.detections.camera_id for view in method.received[1].views} == {"near", "far"}
    for before, after in zip(first, second, strict=True):
        np.testing.assert_array_equal(before.result.positions, after.result.positions)
        np.testing.assert_array_equal(before.result.valid, after.result.valid)
    # A method returning an exchanged camera axis fails at the runner boundary.
    class Reordered(FixedPointMethod):
        def __call__(self, data):
            return replace(super().__call__(data), camera_ids=tuple(reversed(data.camera_ids)))
    with pytest.raises(ValueError, match="selected camera"):
        run_method(Reordered(), inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z")


def test_fractional_offsets_and_motion_never_bridge_invalid_frames():
    inputs = make_inputs()
    valid = np.array([True, False, True, True])
    result = Ball3DResult(inputs.camera_ids, inputs.frame_indices,
        np.array([[1, 2, 1], [0, 0, 0], [3, 2, 1], [4, 2, 1]], np.float32), valid,
        np.tile(valid, (3, 1)), time_offsets_seconds=np.array([.05, 0., 0.], np.float64))
    uv, projected = project_result(inputs, result, "right")
    np.testing.assert_array_equal(projected, [False, False, True, False])
    np.testing.assert_allclose(uv[2], [254.28571428571428, 8.57142857142857])
    motion = motion_samples(result, inputs.fps)
    np.testing.assert_array_equal(motion["step_frames"], [3])
    np.testing.assert_array_equal(motion["step_m"], [1])
    assert len(motion["acceleration_mps2"]) == 0


def test_annotation_oracle_uses_retained_observations_only():
    inputs = make_inputs()
    base = np.array([[211.42857142857142, 8.57142857142857],
                     [122.22222222222223, 55.55555555555556], [160., 187.5]])
    uv = np.tile(base[:, None], (1, 4, 1))
    states: NDArray[np.uint8] = np.ones((3, 4), np.uint8)
    states[0, 1] = 0
    uv[0, 1] = 0
    labels = Ball2DAnnotations(inputs.camera_ids, inputs.frame_indices, inputs.source_size_wh, inputs.fps,
                              uv, states, {"kind": "synthetic_test_fixture"})
    runs = annotation_oracle(inputs, labels, bundle_sha256="x", code_commit="y", code_fingerprint="z")
    np.testing.assert_allclose(runs[0].result.positions, np.tile([1., 2., 1.], (4, 1)), atol=1e-6)
    assert runs[1].held_out_camera == "right" and runs[1].result.valid.all()
    np.testing.assert_array_equal(runs[2].result.valid, [True, False, True, True])
    assert all(run.config["scoring_only"] for run in runs)
