"""Threshold-free, source-pixel candidate recall for single observed balls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CandidateRecallCounts:
    """Additive frame counts; every rate uses observed frames unless named otherwise."""

    frames: int
    observed: int
    recalled_at_1: int
    recalled_at_k: int
    wrong_ranked_above_true: int
    wrong_strictly_higher_score: int

    def report(self) -> dict[str, int | float | None]:
        def rate(count: int, denominator: int) -> float | None:
            return count / denominator if denominator else None

        return {
            **asdict(self),
            "not_in_candidates": self.observed - self.recalled_at_k,
            "recall_at_1": rate(self.recalled_at_1, self.observed),
            "recall_at_k": rate(self.recalled_at_k, self.observed),
            "not_in_candidates_rate": rate(self.observed - self.recalled_at_k, self.observed),
            "wrong_ranked_above_true_rate": rate(self.wrong_ranked_above_true, self.observed),
            "wrong_ranked_above_true_given_recalled": rate(self.wrong_ranked_above_true, self.recalled_at_k),
            "wrong_strictly_higher_score_rate": rate(self.wrong_strictly_higher_score, self.observed),
            "rank_only_due_to_tie": self.wrong_ranked_above_true - self.wrong_strictly_higher_score,
        }


def candidate_recall_counts(
    candidate_xy: NDArray[np.float32], scores: NDArray[np.float32], valid: NDArray[np.bool_],
    target_xy: NDArray[np.float32], observed: NDArray[np.bool_], *, radius_px: float,
) -> CandidateRecallCounts:
    """Use score order, retain decoder order for ties, never gate by confidence.

    Inputs already use source pixels. A hit is distance <= radius. Unknown GT
    is excluded before arithmetic. Invalid padded candidates cannot be hits.
    The ranking error means a correct candidate exists but top-1 is incorrect;
    absent correct candidates are counted separately, not as ranking errors.
    """
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("Candidate scores must have shape (frames, K), K >= 1")
    n = scores.shape[0]
    if (candidate_xy.shape != (*scores.shape, 2) or valid.shape != scores.shape
            or target_xy.shape != (n, 2) or observed.shape != (n,)):
        raise ValueError("Candidate and target shapes must share a frame timeline")
    if valid.dtype != np.bool_ or observed.dtype != np.bool_:
        raise ValueError("Candidate and observed masks must be boolean")
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Source-pixel radius must be positive and finite")
    if (not np.isfinite(candidate_xy[valid]).all() or not np.isfinite(scores[valid]).all()
            or not np.isfinite(target_xy[observed]).all()):
        raise ValueError("Valid candidate/observed target values must be finite")
    if ((scores[valid] < 0) | (scores[valid] > 1)).any():
        raise ValueError("Candidate scores must lie in [0, 1]")
    coords, confidence, mask = candidate_xy[observed], scores[observed], valid[observed]
    ranked_scores = np.where(mask, confidence, -np.inf)
    order = np.argsort(-ranked_scores, axis=1, kind="stable")
    # Float64 distances preserve the inclusive 20-source-pixel boundary.
    distance = np.linalg.norm(coords.astype(np.float64) - target_xy[observed, None].astype(np.float64), axis=-1)
    correct = mask & (distance <= radius_px)
    first_correct = np.take_along_axis(correct, order[:, :1], axis=1).reshape(-1)
    recalled = correct.any(axis=1)
    best_true = np.where(correct, confidence, -np.inf).max(axis=1)
    best_wrong = np.where(mask & ~correct, confidence, -np.inf).max(axis=1)
    return CandidateRecallCounts(
        frames=n, observed=int(observed.sum()), recalled_at_1=int(first_correct.sum()),
        recalled_at_k=int(recalled.sum()), wrong_ranked_above_true=int((recalled & ~first_correct).sum()),
        wrong_strictly_higher_score=int((recalled & (best_wrong > best_true)).sum()),
    )
