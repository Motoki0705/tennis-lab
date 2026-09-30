"""Counterfactual coordinates and label denominators must not bias the side audit."""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def diagnostic(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('court_side_clip000_counterfactuals')


def cache_arrays() -> dict[str, Any]:
    coords = np.array([[0, 0], [1, 1]], np.float32)
    scores = np.array([.25, .75], np.float32)
    return {'frame_index': np.array([0, 1]), 'candidate_coords': coords[:, None],
            'candidate_scores': scores[:, None], 'candidate_valid': np.ones((2, 1), bool),
            'argmax_uv': coords.copy(), 'argmax_score': scores.copy()}


def test_e9_uses_source_grid_not_stored_jpeg_dimensions(diagnostic: Any) -> None:
    uv, scores, valid = diagnostic.source_top1(cache_arrays(), 1920, 1080, 2)
    np.testing.assert_array_equal(uv, [[0, 0], [1919, 1079]])
    np.testing.assert_array_equal(scores, [.25, .75])
    assert valid.all()  # Unthresholded diagnostic must retain a low-scoring top1.


@pytest.mark.parametrize('mismatch', ['frame', 'rank'])
def test_cache_alignment_and_top1_are_not_guessed(diagnostic: Any, mismatch: str) -> None:
    arrays = cache_arrays()
    if mismatch == 'frame':
        arrays['frame_index'] += 1
    else:
        arrays['argmax_uv'][0] = [.5, .5]
    with pytest.raises(AssertionError):
        diagnostic.source_top1(arrays, 1920, 1080, 2)


def test_estimated_or_missing_label_is_not_a_negative(diagnostic: Any) -> None:
    uv = np.array([[0, 0], [100, 100], [500, 500]], np.float32)
    labels = np.zeros_like(uv)
    quality = diagnostic.agreement(uv, np.ones(3, bool), labels, np.array([True, True, False]), 20.)
    assert quality['predicted'] == 3 and quality['both'] == 2
    assert quality['matched'] == 1 and quality['far'] == 1
    assert quality['predicted_without_observed_label'] == 1
    assert quality['recall_at_20px'] == quality['precision_on_observed_labels_at_20px'] == .5
