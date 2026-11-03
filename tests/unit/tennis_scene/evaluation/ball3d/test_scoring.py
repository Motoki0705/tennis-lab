from dataclasses import fields, replace

import numpy as np
import pytest
from numpy.typing import NDArray

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
from tests.support.ball3d_evaluation import (
    make_annotations,
    make_inputs,
    make_scoring_bundle,
)


class FixedPointMethod:
    name = "independent_fixture"
    config = {"fixture": "hand-calculated pinhole projections"}
    requires_court_lines = False

    def __init__(self):
        self.received = []

    def __call__(self, inputs):
        self.received.append(inputs)
        valid = np.array([True, True, True, False])
        xyz = np.tile(np.array([1.0, 2.0, 1.0], np.float32), (4, 1))
        xyz[~valid] = 0
        return Ball3DResult(
            inputs.camera_ids,
            inputs.frame_indices,
            xyz,
            valid,
            np.tile(valid, (len(inputs.views), 1)),
        )


def _labels(inputs):
    # Derived on paper from the three distinct focal lengths and camera centers:
    # right: (160+120*3/7,120-130*6/7); near: (160-170*2/9,120-145*4/9);
    # far: (160,120+180*3/8). These do not call the production projector.
    base = np.array(
        [
            [211.42857142857142, 8.57142857142857],
            [122.22222222222223, 55.55555555555556],
            [160.0, 187.5],
        ]
    )
    uv: NDArray[np.float64] = np.zeros((3, 4, 2), np.float64)
    states: NDArray[np.str_] = np.full((3, 4), "unknown", dtype="U22")
    uv[:, 0] = base + [[3, 4], [5, 12], [7, 24]]  # Euclidean distances 5, 13, 25.
    uv[:2, 2] = base[:2] + [[0, 0], [3, 4]]  # 0, 5; far is occluded.
    uv[:, 3] = base + [[7, 24], [5, 12], [3, 4]]  # Result invalid: never scored.
    states[:, [0, 3]] = "observed"
    states[:2, 2] = "observed"
    states[2, 2] = "occluded"
    states[1, 0] = "estimated"  # Approved 13 px reference; never relabel as observed.
    return make_annotations(inputs, uv, states)


def test_lovo_heldout_pixels_missingness_and_time_splits_match_hand_calculation():
    inputs, method = make_inputs(), FixedPointMethod()
    runs = run_method(
        method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
    )
    labels = _labels(inputs)
    report = score_comparison(
        {method.name: runs},
        make_scoring_bundle(inputs),
        labels,
        config=ScoringConfig(split_frame=2),
    )
    metrics = report["methods"][method.name]
    tune, evaluate, all_frames = [
        metrics["lovo_reprojection_px"][key] for key in ("tuning", "evaluation", "all")
    ]
    assert (tune["count"], evaluate["count"], all_frames["count"]) == (3, 2, 5)
    assert tune["median"] == pytest.approx(13)
    assert tune["p90"] == pytest.approx(22.6)
    assert tune["above_threshold_rate"] == pytest.approx(1 / 3)
    assert evaluate["median"] == pytest.approx(2.5)
    assert evaluate["p90"] == pytest.approx(4.5)
    assert all_frames["median"] == pytest.approx(5)
    assert all_frames["p90"] == pytest.approx(20.2)
    assert all_frames["above_threshold_rate"] == pytest.approx(1 / 5)
    assert metrics["folds"]["all_views"]["evaluation"][
        "scored_label_coverage"
    ] == pytest.approx(2 / 5)
    assert (
        metrics["folds"]["without_far"]["evaluation"]["reprojection_px"]["count"] == 0
    )
    assert metrics["folds"]["all_views"]["all"][
        "gross_reprojection_frame_rate"
    ] == pytest.approx(1 / 2)
    # Change only non-held-out labels; the right-camera fold must be unchanged.
    changed = labels.uv_px.copy()
    changed[1:, 0] = [20, 20]
    other = make_annotations(inputs, changed, labels.reference_kind)
    np.testing.assert_array_equal(
        error_samples(runs[1], make_scoring_bundle(inputs), labels)["error_px"],
        error_samples(runs[1], make_scoring_bundle(inputs), other)["error_px"],
    )
    breakdown = metrics["lovo_reprojection_by_reference_kind"]["tuning"]
    assert breakdown["observed"]["count"] == 2
    assert breakdown["observed"]["median"] == pytest.approx(15)
    assert breakdown["estimated"]["count"] == 1
    assert breakdown["estimated"]["median"] == pytest.approx(13)
    assert breakdown["all"]["label_count"] == 3


def test_method_runner_excludes_labels_reference_and_heldout_camera():
    inputs, method = make_inputs(), FixedPointMethod()
    first = run_method(
        method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
    )
    score_comparison(
        {method.name: first},
        make_scoring_bundle(inputs),
        _labels(inputs),
        config=ScoringConfig(split_frame=2),
    )
    second = run_method(
        method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
    )
    assert [row.camera_ids for row in method.received[:4]] == [
        ("right", "near", "far"),
        ("near", "far"),
        ("right", "far"),
        ("right", "near"),
    ]
    assert {field.name for field in fields(method.received[1])} == {
        "clip_id",
        "frame_indices",
        "fps",
        "source_size_wh",
        "views",
    }
    assert {view.detections.camera_id for view in method.received[1].views} == {
        "near",
        "far",
    }
    assert [view.court.camera_id for view in method.received[1].views] == [
        "near",
        "far",
    ]
    # The excluded right view's distinct raw (21.25,50.5) point is unavailable.
    assert [
        view.court.keypoints_px[0, 0].tolist() for view in method.received[1].views
    ] == [[68.25, 79.5], [115.25, 108.5]]
    for before, after in zip(first, second, strict=True):
        np.testing.assert_array_equal(before.result.positions, after.result.positions)
        np.testing.assert_array_equal(before.result.valid, after.result.valid)

    # A method returning an exchanged camera axis fails at the runner boundary.
    class Reordered(FixedPointMethod):
        def __call__(self, data):
            return replace(
                super().__call__(data), camera_ids=tuple(reversed(data.camera_ids))
            )

    with pytest.raises(ValueError, match="selected camera"):
        run_method(
            Reordered(),
            inputs,
            bundle_sha256="x",
            code_commit="y",
            code_fingerprint="z",
        )


def test_fractional_offsets_and_motion_never_bridge_invalid_frames():
    inputs = make_inputs()
    valid = np.array([True, False, True, True])
    result = Ball3DResult(
        inputs.camera_ids,
        inputs.frame_indices,
        np.array([[1, 2, 1], [0, 0, 0], [3, 2, 1], [4, 2, 1]], np.float32),
        valid,
        np.tile(valid, (3, 1)),
        time_offsets_seconds=np.array([0.05, 0.0, 0.0], np.float64),
    )
    uv, projected = project_result(inputs, result, "right")
    np.testing.assert_array_equal(projected, [False, False, True, False])
    np.testing.assert_allclose(uv[2], [254.28571428571428, 8.57142857142857])
    motion = motion_samples(result, inputs.fps)
    np.testing.assert_array_equal(motion["step_frames"], [3])
    np.testing.assert_array_equal(motion["step_m"], [1])
    assert len(motion["acceleration_mps2"]) == 0


def test_annotation_oracle_uses_retained_observations_only():
    inputs = make_inputs()
    base = np.array(
        [
            [211.42857142857142, 8.57142857142857],
            [122.22222222222223, 55.55555555555556],
            [160.0, 187.5],
        ]
    )
    uv = np.tile(base[:, None], (1, 4, 1))
    states: NDArray[np.str_] = np.full((3, 4), "observed", dtype="U22")
    states[0, 1] = "excluded_extrapolation"
    states[1, 2] = "estimated"
    uv[0, 1] = 0
    labels = make_annotations(inputs, uv, states)
    runs = annotation_oracle(
        make_scoring_bundle(inputs), labels, code_commit="y", code_fingerprint="z"
    )
    np.testing.assert_allclose(
        runs[0].result.positions, np.tile([1.0, 2.0, 1.0], (4, 1)), atol=1e-6
    )
    assert runs[1].held_out_camera == "right" and runs[1].result.valid.all()
    np.testing.assert_array_equal(runs[2].result.valid, [True, False, True, True])
    assert all(run.config["scoring_only"] for run in runs)


def test_camera_event_rows_use_union_windows_without_multiplying_residuals():
    inputs, method = make_inputs(), FixedPointMethod()
    labels = _labels(inputs)
    rows = (
        {
            "camera_id": "right",
            "frame_index": 0,
            "event": "bounce",
            "note": "contact ±1 frame",
        },
        {
            "camera_id": "near",
            "frame_index": 0,
            "event": "bounce",
            "note": "same event",
        },
        {
            "camera_id": "far",
            "frame_index": 1,
            "event": "bounce",
            "note": "one frame later",
        },
    )
    labels = make_annotations(inputs, labels.uv_px, labels.reference_kind, events=rows)
    runs = run_method(
        method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
    )
    report = score_comparison(
        {method.name: runs},
        make_scoring_bundle(inputs),
        labels,
        config=ScoringConfig(split_frame=2, event_radius_frames=0),
    )
    bounce = report["methods"][method.name]["folds"]["all_views"]["tuning"][
        "event_reprojection_px"
    ]["bounce"]
    assert bounce["source_row_count"] == 3 and bounce["physical_event_count"] is None
    assert bounce["source_rows"] == list(rows)
    assert bounce["error_px"]["count"] == 3
    assert bounce["error_px"]["median"] == pytest.approx(13)
    assert bounce["by_reference_kind"]["estimated"]["count"] == 1


def test_motion_outliers_respect_gaps_and_split_boundaries():
    inputs = make_inputs(frames=6)

    class MotionMethod:
        name, config = "hand_motion", {}
        requires_court_lines = False

        def __call__(self, data):
            valid = np.array([True, True, True, False, True, True])
            xyz = np.array(
                [[0, 0, 1], [1, 0, 1], [3, 0, 1], [0, 0, 0], [5, 0, 1], [9, 0, 1]],
                np.float32,
            )
            return Ball3DResult(
                data.camera_ids,
                data.frame_indices,
                xyz,
                valid,
                np.tile(valid, (len(data.views), 1)),
            )

    runs = run_method(
        MotionMethod(), inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
    )
    report = score_comparison(
        {"hand_motion": runs},
        make_scoring_bundle(inputs),
        None,
        config=ScoringConfig(split_frame=2, max_speed_mps=30, max_acceleration_mps2=90),
    )
    folds = report["methods"]["hand_motion"]["folds"]["all_views"]
    # At 10 fps the valid steps are 1,2,4 m, with one 100 m/s² acceleration.
    assert folds["all"]["adjacent_step_m"]["median"] == pytest.approx(2)
    assert folds["all"]["speed_mps"]["above_threshold_rate"] == pytest.approx(1 / 3)
    assert folds["all"]["acceleration_mps2"]["above_threshold_rate"] == 1
    assert folds["tuning"]["acceleration_mps2"]["count"] == 0
    assert folds["evaluation"]["acceleration_mps2"]["count"] == 0
    assert folds["evaluation"]["adjacent_step_m"]["count"] == 1
    assert folds["evaluation"]["speed_mps"]["above_threshold_rate"] == 1


@pytest.mark.parametrize("mismatch", ["bundle_digest", "frame_count"])
def test_scoring_rejects_results_from_other_bundle_or_timeline(mismatch):
    inputs, method = make_inputs(), FixedPointMethod()
    bundle, labels = make_scoring_bundle(inputs), _labels(inputs)
    runs = list(
        run_method(
            method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
        )
    )
    if mismatch == "bundle_digest":
        runs[0] = replace(runs[0], bundle_sha256="another_bundle")
    else:
        old = runs[0].result
        shorter = Ball3DResult(
            old.camera_ids,
            old.frame_indices[:3],
            old.positions[:3],
            old.valid[:3],
            old.used_views[:, :3],
        )
        runs[0] = replace(runs[0], result=shorter)
    with pytest.raises(ValueError, match="bundle digest|frame axis"):
        score_comparison(
            {method.name: runs}, bundle, labels, config=ScoringConfig(split_frame=2)
        )
    with pytest.raises(ValueError, match="bundle digest|frame axis"):
        error_samples(runs[0], bundle, labels)


@pytest.mark.parametrize("mismatch", ["clip_id", "frame_count", "video_hash"])
def test_scoring_rejects_reference_identity_mismatch(mismatch):
    inputs, method = make_inputs(), FixedPointMethod()
    bundle, labels = make_scoring_bundle(inputs), _labels(inputs)
    identity = labels.reference.identity
    if mismatch == "clip_id":
        identity = replace(identity, clip_id="different_clip")
    elif mismatch == "frame_count":
        identity = replace(identity, num_frames=3)
    else:
        identity = replace(
            identity, video_sha256={**identity.video_sha256, "near": "f" * 64}
        )
    changed = replace(labels, reference=replace(labels.reference, identity=identity))
    runs = run_method(
        method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
    )
    with pytest.raises(ValueError, match="axes|video hash"):
        score_comparison(
            {method.name: runs}, bundle, changed, config=ScoringConfig(split_frame=2)
        )
    with pytest.raises(ValueError, match="axes|video hash"):
        error_samples(runs[0], bundle, changed)


def test_common_lovo_intersection_compares_identical_camera_frame_pairs():
    inputs, first = make_inputs(), FixedPointMethod()

    class MissingRightPoint(FixedPointMethod):
        name = "one_missing_point"

        def __call__(self, data):
            result = super().__call__(data)
            if data.camera_ids == ("near", "far"):
                valid, positions, used = (
                    result.valid.copy(),
                    result.positions.copy(),
                    result.used_views.copy(),
                )
                valid[0], positions[0], used[:, 0] = False, 0, False
                result = replace(
                    result, valid=valid, positions=positions, used_views=used
                )
            return result

    methods = (first, MissingRightPoint())
    runs = {
        method.name: run_method(
            method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
        )
        for method in methods
    }
    report = score_comparison(
        runs,
        make_scoring_bundle(inputs),
        _labels(inputs),
        config=ScoringConfig(split_frame=2),
    )
    assert report["common_lovo_cohort"]["camera_frame_pairs"] == [
        ["right", 2],
        ["near", 0],
        ["near", 2],
        ["far", 0],
    ]
    assert report["methods"][first.name]["lovo_reprojection_px"]["all"]["count"] == 5
    for name in runs:
        common = report["methods"][name]["common_lovo_reprojection_px"]["all"]
        assert common["count"] == 4 and common["label_count"] == 8
        assert common["median"] == pytest.approx(9)  # 0,5,13,25
        assert common["p90"] == pytest.approx(21.4)
        assert common["scored_label_coverage"] == 0.5


def test_runner_stops_before_method_when_required_line_observations_are_missing():
    inputs = make_inputs()
    views = list(inputs.views)
    views[1] = replace(
        views[1],
        court=replace(
            views[1].court,
            lines=None,
            line_missing_reason="fixture did not record LINE",
        ),
    )
    inputs = replace(inputs, views=tuple(views))

    class NeedsLines(FixedPointMethod):
        requires_court_lines = True

    method = NeedsLines()
    with pytest.raises(ValueError, match="near.*LINE observations unavailable"):
        run_method(
            method, inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z"
        )
    assert method.received == []
