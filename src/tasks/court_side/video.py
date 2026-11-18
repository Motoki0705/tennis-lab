"""Pool scores from distinct clip timelines using the original per-frame weights."""
from __future__ import annotations

from collections.abc import Mapping
from math import fsum

import numpy as np

from .hypothesis import BallSideEvidence, HypothesisScore


def pool_side_evidence(clips: Mapping[str, BallSideEvidence]) -> BallSideEvidence:
    """Sum pair counts and weight each assignment by its distinct multi-view frames.

    Motion filtering and view matching have already happened within each clip.
    A zero-frame clip contributes neither a cost nor a vote.
    """
    if not clips:
        raise ValueError("Shared court side requires clip evidence")
    ordered = [clips[key] for key in sorted(clips)]
    first = ordered[0]
    scores: dict[tuple[bool, ...], list[HypothesisScore]] = {}
    for clip in ordered:
        if (clip.camera_ids, clip.reference_camera) != (first.camera_ids, first.reference_camera):
            raise ValueError("Shared court side calibration cameras/reference differ between clips")
        for hypothesis in clip.hypotheses:
            scores.setdefault(hypothesis.view_half_turns, []).append(hypothesis)
    frames = sum(c.frames for c in ordered)
    pooled = [
        HypothesisScore(turns, fsum(s.cost * s.frames for s in rows) / frames,
                        fsum(s.support * s.frames for s in rows) / frames, frames)
        for turns, rows in scores.items()
    ]
    return BallSideEvidence(first.camera_ids, first.reference_camera, frames,
                            np.sum([c.pair_frames for c in ordered], axis=0),
                            tuple(sorted(pooled, key=lambda h: (h.cost, h.view_half_turns))))
