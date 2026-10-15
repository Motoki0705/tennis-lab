from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.contracts import DetectionFeatures, ScoreOrigin
from src.tasks.person_tracking.methods import build_tracker
from src.tasks.person_tracking.sequence import DatasetTrackingConfig
from src.tasks.person_tracking.strongsort import (
    MotionState,
    StrongSort,
    StrongSortConfig,
)


def mixed(frame: int = 0) -> DetectionFeatures:
    return DetectionFeatures(
        frame=frame,
        rows=np.array([frame * 10 + 2, frame * 10 + 7], np.int64),
        boxes=np.array([[0, 0, 20, 60], [80, 0, 100, 60]], np.float32),
        scores=np.array([0.9, 0], np.float32),
        poses=np.zeros((2, 17, 3), np.float32),
        embeddings=np.eye(2, dtype=np.float32),
        appearance_valid=np.ones(2, bool),
        score_origin=np.array([ScoreOrigin.MODEL, ScoreOrigin.MISSING], np.uint8),
    )


def test_missing_confidence_is_distinct_from_a_real_zero_score() -> None:
    features = mixed()
    assert features.score_at(0) == pytest.approx(0.9)
    assert features.score_at(1) is None
    with pytest.raises(ValueError, match="requires a score"):
        features.require_scores()
    legacy = replace(features, score_origin=None)
    assert legacy.score_at(1) == 0.0
    assert legacy.score_origins().tolist() == [ScoreOrigin.LEGACY] * 2
    np.testing.assert_array_equal(legacy.require_scores(), features.scores)
    with pytest.raises(IndexError):
        features.score_at(-1)


@pytest.mark.parametrize(
    "origin",
    [np.array([2], np.uint8), np.array([2, 0], np.int64), np.array([2, 3], np.uint8)],
)
def test_missing_score_mask_rejects_wrong_axis_dtype_and_unknown_code(
    origin: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="Score origins"):
        replace(mixed(), score_origin=origin)


def test_corrected_observation_cannot_keep_a_model_score() -> None:
    with pytest.raises(ValueError, match="copied or invented confidence"):
        replace(mixed(), scores=np.array([0.9, 0.95], np.float32))


@pytest.mark.parametrize("method", ["botsort_pose", "deep_ocsort_pose"])
def test_score_threshold_methods_reject_missing_scores_before_mutating_state(
    method: str,
) -> None:
    tracker = build_tracker(method, fps=30.0)
    with pytest.raises(ValueError, match="requires a score"):
        tracker.update(mixed())
    # The same frame can still be processed after an explicit, scored replacement.
    result = tracker.update(
        replace(mixed(), scores=np.array([0.9, 0.9], np.float32), score_origin=None)
    )
    assert result.detection_rows.tolist() == [2, 7]


def test_unscored_box_tracks_with_explicit_fixed_noise_and_keeps_real_rows() -> None:
    config = StrongSortConfig(unscored_measurement_noise_scale=2.0)
    unknown = StrongSort(config)
    zero = StrongSort(config)
    for frame in range(3):
        observations = mixed(frame)
        a = unknown.update(observations)
        b = zero.update(replace(observations, score_origin=None))
    assert a.track_ids.tolist() == b.track_ids.tolist() == [1, 2]
    assert a.detection_rows.tolist() == b.detection_rows.tolist() == [22, 27]
    # The true model-scored row retains its previous numerical behavior.
    np.testing.assert_array_equal(
        unknown.tracks[0].motion.mean, zero.tracks[0].motion.mean
    )
    np.testing.assert_array_equal(
        unknown.tracks[0].motion.covariance, zero.tracks[0].motion.covariance
    )
    # An unscored observation follows the configured noise rule, not score=0.
    assert (
        unknown.tracks[1].motion.covariance[0, 0]
        > zero.tracks[1].motion.covariance[0, 0]
    )
    assert (
        DatasetTrackingConfig().identity()["online_config"][
            "unscored_measurement_noise_scale"
        ]
        == 1.0
    )


@pytest.mark.parametrize(
    "score,expected_x,expected_variance",
    [
        (0.1, 14.396984925501108, 7.1231155675181625),
        (0.9, 14.924953078060785, 0.886491765406987),
    ],
)
def test_scored_motion_matches_frozen_prechange_numerical_results(
    score: float, expected_x: float, expected_variance: float
) -> None:
    # Recorded from d734b1308 MotionState using float32 detector scores, CPU only.
    box = np.array([0, 0, 20, 60], np.float32)
    state = MotionState.initiate(box)
    state.predict()
    state.update(box + 5, measurement_noise_scale=1 - float(np.float32(score)))
    assert state.mean[0] == pytest.approx(expected_x, rel=1e-13)
    assert state.covariance[0, 0] == pytest.approx(expected_variance, rel=1e-13)


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_invalid_unscored_noise_rule_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="noise scale"):
        StrongSortConfig(unscored_measurement_noise_scale=scale)


def test_archive_preserves_mixed_score_origins_and_reads_v2_as_legacy(
    tmp_path: Path,
) -> None:
    original = mixed()
    path = tmp_path / "mixed.npz"
    save_features(path, [original], {"annotation_revision": "a" * 64})
    loaded, provenance = load_features(path)
    assert loaded[0].score_at(1) is None
    assert loaded[0].score_origins().tolist() == [
        ScoreOrigin.MODEL,
        ScoreOrigin.MISSING,
    ]
    np.testing.assert_array_equal(loaded[0].scores, original.scores)
    assert provenance == {"annotation_revision": "a" * 64}
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads(arrays["metadata"].item())
    assert metadata["schema"] == "person_detection_features_v3"
    metadata["schema"] = "person_detection_features_v2"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    del arrays["score_origin"]
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, **arrays)
    loaded, _ = load_features(legacy)
    assert loaded[0].score_at(1) == 0.0
    assert loaded[0].score_origins().tolist() == [ScoreOrigin.LEGACY] * 2
    metadata["schema"] = "person_detection_features_v3"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    broken = tmp_path / "missing-origin.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(ValueError, match="fields differ"):
        load_features(broken)
