"""Filtering logic for pseudo-label reliability."""

from __future__ import annotations

from src.ball_detection.data.type import ConfidenceRecord


class QualityFilter:
    """Select frames eligible for pseudo-label supervision."""

    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = float(min_confidence)

    def keep_indices(self, confidence: dict[int, ConfidenceRecord]) -> set[int]:
        return {
            idx
            for idx, c in confidence.items()
            if c.confidence >= self.min_confidence
        }
