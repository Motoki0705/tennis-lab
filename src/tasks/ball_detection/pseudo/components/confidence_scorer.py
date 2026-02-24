"""Confidence scoring from ensemble outputs and event context."""

from __future__ import annotations

from src.tasks.ball_detection.data.type import ConfidenceRecord, EventRecord


class ConfidenceScorer:
    """Compute frame-wise confidence used for pseudo-label acceptance."""

    def __init__(self, event_penalty: float = 0.15) -> None:
        self.event_penalty = float(event_penalty)

    def score(
        self,
        *,
        visibility: list[bool],
        detector_scores: list[float],
        events: dict[int, EventRecord],
    ) -> dict[int, ConfidenceRecord]:
        conf: dict[int, ConfidenceRecord] = {}
        for i, (vis, det_score) in enumerate(zip(visibility, detector_scores, strict=True)):
            base = float(det_score) if vis else 0.0
            evt = events.get(i)
            if evt is not None and max(evt.shot_prob, evt.bounce_prob) >= 0.5:
                base = max(0.0, base - self.event_penalty)
            conf[i] = ConfidenceRecord(frame_index=i, confidence=base, source="ensemble")
        return conf
