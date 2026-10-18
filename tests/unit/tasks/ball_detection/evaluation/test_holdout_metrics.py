"""Missing predictions and uncertain references must not improve reported errors."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.teacher_masks import EXPLICIT_MASK_SCHEMA, LEGACY
from src.tasks.ball_detection.evaluation.holdout_inference import FramePredictions
from src.tasks.ball_detection.evaluation.holdout_metrics import (
    HoldoutReferences,
    clip_references,
    summarize_holdout,
    wrist_distances,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def references() -> HoldoutReferences:
    return HoldoutReferences(
        np.arange(5, dtype=np.int64), np.full(5, "clip"), np.full(5, "cam0"),
        np.arange(5, dtype=np.int64), np.ones(5, bool),
        np.asarray(["observed"] * 3 + ["occlusion_estimated", "unresolved"]),
        np.asarray([False, False, False, True, True]),
        np.asarray([[0, 0]] * 4 + [[np.nan, np.nan]], np.float32),
        np.asarray([5, 200, np.nan, np.nan, np.nan], np.float32),
    )


def predictions() -> FramePredictions:
    return FramePredictions(
        np.asarray([[0, 0], [100, 0], [10, 0], [200, 0], [500, 0]], np.float32),
        np.asarray([0.5, 0.9, 0.1, 0.7, 0.9], np.float32),
        np.zeros((5, 2, 2), np.float32), np.full((5, 2), 0.1, np.float32),
        np.ones((5, 2), bool), np.zeros(5, np.int64), np.arange(5, dtype=np.int64),
    )


def summarize(ref=None, pred=None, **kwargs):
    return summarize_holdout(
        ref if ref is not None else references(), pred if pred is not None else predictions(),
        score_threshold=0.5, distance_px=20, near_wrist_px=100, **kwargs,
    )


def test_large_errors_and_misses_remain_visible():
    result = summarize()
    overall = result["overall_observed"]
    assert overall["reference_frames"] == 3
    assert overall["recall"] == pytest.approx(1 / 3)
    assert overall["missing_reference_frames"] == 1
    assert overall["wrong_reference_frames"] == 1
    assert overall["accepted_p95_px"] == pytest.approx(95)
    assert overall["raw_argmax_p95_px"] == pytest.approx(91)
    assert overall["topk_recall_unthresholded"] == 1
    assert result["point_kind/occlusion_estimated"]["raw_argmax_p95_px"] == 200
    unknown = result["point_kind/unresolved"]
    assert unknown["detected_without_reference_frames"] == 1
    assert unknown["reference_frames"] == 0
    assert unknown["recall"] is None
    assert unknown["accepted_p95_px"] is None
    assert result["wrist/near_wrist"]["reference_frames"] == 1
    assert result["wrist/flight"]["reference_frames"] == 1
    assert result["wrist/unknown"]["reference_frames"] == 1


def test_all_misses_have_null_accepted_quantile_and_zero_recall():
    pred = replace(predictions(), score=np.zeros(5, np.float32), candidate_valid=np.zeros((5, 2), bool))
    overall = summarize(pred=pred)["overall_observed"]
    assert overall["missing_reference_frames"] == 3
    assert overall["recall"] == overall["topk_recall_unthresholded"] == 0
    assert overall["accepted_p95_px"] is None
    assert overall["raw_argmax_p95_px"] == pytest.approx(91)


def test_unreviewed_observed_is_not_scored():
    ref = replace(references(), annotated=np.zeros(5, bool))
    overall = summarize(ref=ref)["overall_observed"]
    assert overall["reference_frames"] == 0
    assert overall["recall"] is None


def test_frame_duplication_and_missing_located_coordinates_are_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        replace(references(), row=np.zeros(5, np.int64))
    with pytest.raises(ValueError, match="located references"):
        replace(references(), uv=np.full((5, 2), np.nan, np.float32))
    with pytest.raises(ValueError, match="finite float32"):
        replace(predictions(), score=np.full(5, np.nan, np.float32))


def test_wrist_missing_support_does_not_become_flight():
    balls = np.asarray([[0, 0], [0, 0], [np.nan, np.nan]], np.float32)
    wrists = np.asarray([[[3, 4], [100, 100]], [[3, 4], [0, 0]], [[3, 4], [0, 0]]], np.float32)
    valid = np.asarray([[True, False], [False, False], [True, True]])
    result = wrist_distances(balls, wrists, valid)
    np.testing.assert_allclose(result, [5, np.nan, np.nan], equal_nan=True)
    assert np.isnan(wrist_distances(balls, np.zeros((3, 0, 2), np.float32), np.zeros((3, 0), bool))).all()


def test_invalid_wrist_thresholds_and_axes_are_rejected():
    with pytest.raises(ValueError, match="Wrist coordinates"):
        wrist_distances(np.zeros((2, 2), np.float32), np.zeros((1, 2, 2), np.float32), np.ones((1, 2), bool))
    with pytest.raises(ValueError, match="thresholds"):
        summarize_holdout(references(), predictions(), score_threshold=0.5, distance_px=float("nan"), near_wrist_px=100)


def test_adopted_mask_excludes_untrusted_coordinates_from_errors_and_recall():
    ref = replace(references(), coordinate_mask=np.asarray([1, 0, 1, 0, 0], np.uint8))
    result = summarize(ref=ref)
    overall = result["overall_observed"]
    assert overall["reference_frames"] == 2
    assert overall["recall"] == 0.5
    assert overall["wrong_reference_frames"] == 0
    assert overall["raw_argmax_p95_px"] == pytest.approx(9.5)
    # Counts and points still describe the review, including excluded references.
    assert result["point_kind/observed"]["frames"] == 3
    assert result["point_kind/observed"]["without_reference_frames"] == 1
    assert result["point_kind/occlusion_estimated"]["reference_frames"] == 0
    assert result["point_kind/occlusion_estimated"]["raw_argmax_p95_px"] is None
    np.testing.assert_array_equal(ref.uv, references().uv)


def test_explicit_legacy_mask_preserves_old_reference_metrics():
    assert summarize(ref=replace(references(), coordinate_mask=np.full(5, LEGACY, np.uint8))) == summarize()


@pytest.mark.parametrize("mask", [
    np.zeros(5, np.int32), np.zeros(4, np.uint8),
    np.asarray([1, 2, 1, 0, 0], np.uint8), np.ones(5, np.uint8),
])
def test_invalid_holdout_teacher_masks_fail_explicitly(mask):
    with pytest.raises(ValueError, match="coordinate mask|reviewed observed"):
        replace(references(), coordinate_mask=mask)


def test_clip_reference_export_carries_exact_adoption_mask(tmp_path: Path):
    root = write_store_clip(tmp_path / "store", "clip", [frame(0, ball()), frame(1, ball())], source="meiji")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["clips"][0]["camera_id"] = "cam0"
    metadata_path.write_text(json.dumps(metadata))
    legacy = BallFrameStore(root)
    reference = clip_references(legacy, legacy.clips[0])
    np.testing.assert_array_equal(reference.coordinate_mask, [LEGACY, LEGACY])
    metadata["schema_version"] = EXPLICIT_MASK_SCHEMA
    metadata_path.write_text(json.dumps(metadata))
    with np.load(root / "index.npz", allow_pickle=False) as archive:
        columns = dict(archive)
    columns.update(teacher_heatmap=np.asarray([0, 2], np.uint8), teacher_coordinate=np.asarray([0, 1], np.uint8))
    np.savez(root / "index.npz", **columns)
    adopted = BallFrameStore(root)
    reference = clip_references(adopted, adopted.clips[0])
    np.testing.assert_array_equal(reference.coordinate_mask, [0, 1])
    assert reference.annotated.tolist() == [True, True]
    assert reference.point_kind.tolist() == ["observed", "observed"]
    assert reference.uv.tolist() == [[10.0, 20.0], [10.0, 20.0]]
