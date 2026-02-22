"""Event detection wrapper used during pseudo-label generation."""

from __future__ import annotations

from src.ball_detection.data.type import EventRecord
from src.event_detection.inference import UVEventPredictor


class EventTagger:
    """Infer per-frame event probabilities on refined trajectories."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        self.predictor: UVEventPredictor | None = None
        if checkpoint_path:
            self.predictor = UVEventPredictor.load_from_checkpoint(checkpoint_path, device=device)

    def tag(self, ball_uv, court_kp=None, *, ball_vis, ball_mask, court_vis=None) -> dict[int, EventRecord]:
        """Return frame-indexed event probabilities."""
        if self.predictor is None:
            return {}

        out = self.predictor.predict(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
        )
        probs = out["event_probs"]  # [B, T, E]
        if probs.dim() == 3:
            probs = probs[0]
        records: dict[int, EventRecord] = {}
        for i in range(probs.shape[0]):
            shot_prob = float(probs[i, 0]) if probs.shape[1] > 0 else 0.0
            bounce_prob = float(probs[i, 1]) if probs.shape[1] > 1 else 0.0
            records[i] = EventRecord(frame_index=i, shot_prob=shot_prob, bounce_prob=bounce_prob)
        return records
