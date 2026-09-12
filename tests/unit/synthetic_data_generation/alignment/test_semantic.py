"""Regression tests for camera-invariant line identity and paired placement."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.semantic import (
    SemanticRasterObjective,
    refine_placement,
    transform_segments,
    typed_court_segments,
)
from src.tasks.court_detection.inference.semantic_lines import (
    merge_camera_line_probabilities,
)
from src.tasks.court_detection.target_schemas import SEMANTIC_LINE_CHANNEL_NAMES


def test_opposite_camera_channel_permutation_preserves_all_line_types() -> None:
    rng = np.random.default_rng(880)
    probabilities = rng.uniform(size=(12, 5, 8)).astype(np.float32)
    probabilities /= probabilities.sum(axis=0)
    reversed_view = probabilities[[0, 2, 1, 4, 3, 6, 5, 8, 7, 9, 11, 10]]
    actual = merge_camera_line_probabilities(
        probabilities, channel_names=SEMANTIC_LINE_CHANNEL_NAMES
    )
    reversed_actual = merge_camera_line_probabilities(
        reversed_view, channel_names=SEMANTIC_LINE_CHANNEL_NAMES
    )
    np.testing.assert_array_equal(actual, reversed_actual)
    np.testing.assert_allclose(actual.sum(axis=0), 1, atol=2e-7)
    np.testing.assert_array_equal(actual[1], probabilities[1] + probabilities[2])


def test_line_merging_rejects_logits_and_wrong_semantics() -> None:
    probabilities: NDArray[np.float32] = np.full((12, 4, 5), 1 / 12, dtype=np.float32)
    with pytest.raises(ValueError, match="channel order"):
        merge_camera_line_probabilities(
            probabilities, channel_names=SEMANTIC_LINE_CHANNEL_NAMES[::-1]
        )
    with pytest.raises(ValueError, match="summing to one"):
        merge_camera_line_probabilities(
            probabilities * 0.5, channel_names=SEMANTIC_LINE_CHANNEL_NAMES
        )
    probabilities[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        merge_camera_line_probabilities(
            probabilities, channel_names=SEMANTIC_LINE_CHANNEL_NAMES
        )


def _objective() -> SemanticRasterObjective:
    spacing = 0.1
    rasters: NDArray[np.float32] = np.zeros((6, 321, 201), dtype=np.float32)
    for kind, segment in typed_court_segments():
        pixels = np.round((segment + [10, 16]) / spacing).astype(np.int32)
        cv2.line(rasters[kind], tuple(pixels[0]), tuple(pixels[1]), 1, 2)
    return SemanticRasterObjective(rasters, (-10, 10, -16, 16), spacing)


def test_semantic_objective_rejects_wrong_type_even_when_binary_identical() -> None:
    objective = _objective()
    wrong = SemanticRasterObjective(
        objective.rasters[[3, 2, 1, 0, 5, 4]], objective.bounds_uv, objective.spacing
    )
    initial = np.zeros(3)
    assert objective.score(initial) > wrong.score(initial) + 0.5
    np.testing.assert_array_equal(
        objective.rasters.sum(axis=0), wrong.rasters.sum(axis=0)
    )


def test_placement_recovers_translation_and_yaw_on_typed_evidence() -> None:
    objective = _objective()
    initial = np.asarray([0.5, -0.4, 0.04])
    fitted = refine_placement(
        objective,
        initial,
        translation_radius_metres=1.0,
        yaw_radius_radians=0.10,
        smoothing_metres=0.1,
        seed=880,
        maximum_iterations=200,
    )
    np.testing.assert_allclose(fitted[:2], 0, atol=0.08)
    assert abs(fitted[2]) < 0.008
    assert objective.score(fitted) > objective.score(initial) + 0.3


def test_out_of_bounds_has_no_support_and_empty_evidence_fails() -> None:
    objective = _objective()
    assert objective.score(np.asarray([100.0, 100.0, 0.0])) == 0
    empty = SemanticRasterObjective(
        np.zeros_like(objective.rasters), objective.bounds_uv, objective.spacing
    )
    with pytest.raises(ValueError, match="without foreground"):
        refine_placement(
            empty,
            np.zeros(3),
            translation_radius_metres=1,
            yaw_radius_radians=0.1,
            smoothing_metres=0.1,
            seed=1,
            maximum_iterations=1,
        )


def test_center_marks_are_short_inward_segments() -> None:
    segments = typed_court_segments()
    assert {kind for kind, _ in segments} == set(range(6))
    for kind, segment in segments:
        if kind == 5:
            assert abs(segment[1, 1]) < abs(segment[0, 1])
    transformed = transform_segments(np.asarray([3.0, 4.0, np.pi]))
    for (_, segment), actual in zip(segments, transformed, strict=True):
        np.testing.assert_allclose(actual, -segment + [3, 4], atol=1e-12)
