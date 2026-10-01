from __future__ import annotations

import numpy as np
import pytest

from src.tasks.player_association.appearance.affinity import (
    AppearanceAffinityConfig,
    appearance_score,
    fit_cosine_log_likelihood_ratio,
    segment_embedding,
)


def test_segment_embedding_averages_only_the_samples_inside_the_segment() -> None:
    frames = np.array([0, 10, 20, 30])
    embeddings = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], np.float32)
    np.testing.assert_allclose(segment_embedding(frames, embeddings, 0, 15), [1., 0.])
    np.testing.assert_allclose(segment_embedding(frames, embeddings, 5, 25), [np.sqrt(.5), np.sqrt(.5)])
    assert segment_embedding(frames, embeddings, 31, 40) is None
    assert segment_embedding(np.zeros(0, np.int64), np.zeros((0, 0), np.float32), 0, 10) is None


def test_score_is_linear_in_cosine_and_clipped() -> None:
    config = AppearanceAffinityConfig("e", slope=20., center=.85, max_abs_score=4.)
    assert appearance_score(.85, config) == 0.
    assert appearance_score(.9, config) == pytest.approx(1.)
    assert appearance_score(.3, config) == -4.


def test_balanced_fit_centres_between_equal_variance_classes_regardless_of_their_counts() -> None:
    rng = np.random.default_rng(1)
    positive = rng.normal(.9, .03, 400)
    negative = rng.normal(.78, .03, 4000)  # ten times more negatives must not move the centre
    slope, center = fit_cosine_log_likelihood_ratio(positive, negative)
    assert center == pytest.approx(.84, abs=.005)
    # For equal-variance Gaussians the log-likelihood-ratio slope is (mu1 - mu0) / sigma^2.
    assert slope == pytest.approx(.12 / .03 ** 2, rel=.15)
    with pytest.raises(ValueError, match="at least two"):
        fit_cosine_log_likelihood_ratio(positive[:1], negative)
