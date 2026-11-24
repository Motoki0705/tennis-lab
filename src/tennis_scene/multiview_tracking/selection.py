"""Rank real 3D trajectories within the doubles-width court region."""

from __future__ import annotations

from math import ceil

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_tracking.schema import (
    PlayerDecision,
    PlayerSelection,
    SelectionReason,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH, HALF_SINGLES_WIDTH

COURT_HALF_X_M = HALF_DOUBLES_WIDTH  # 5.485
COURT_HALF_Y_M = HALF_LENGTH + 5.0  # 16.885
MIN_RESIDENCE_FRACTION = 0.25
MOTION_LAG_SECONDS = 0.25
# Motion can improve the rank by up to ten percent of that side's residence.
MAX_MOTION_BONUS = 0.10
MOTION_AT_FULL_BONUS_M = 0.5


def _court_motion(
    frames: NDArray[np.int64], xy: NDArray[np.float64], lag: int
) -> tuple[float, int]:
    if len(frames) < 2:
        return 0.0, 0
    later = np.minimum(np.searchsorted(frames, frames + lag), len(frames) - 1)
    available = (frames[later] - frames >= lag) & (frames[later] - frames <= 1.5 * lag)
    # Normalize modestly uneven sampling to displacement per nominal lag.
    delta = frames[later[available]] - frames[available]
    displacement = np.linalg.norm(xy[later[available]] - xy[available], axis=1)
    if not len(delta):
        return 0.0, 0
    return float(np.quantile(displacement * lag / delta, 0.75)), len(delta)


def select_players(
    frames: NDArray[np.int64],
    track_ids: NDArray[np.int64],
    torso_m: NDArray[np.float64],
    *,
    num_frames: int,
    fps: float,
    players_per_side: int,
    player_candidates: set[int] | None = None,
) -> PlayerSelection:
    """Count only reconstructed observations; interpolated samples never enter."""
    minimum_frames = ceil(MIN_RESIDENCE_FRACTION * num_frames)
    lag = max(1, round(MOTION_LAG_SECONDS * fps))
    x, y = torso_m[:, :2].astype(np.float32).T
    region = (np.abs(x) <= COURT_HALF_X_M) & (np.abs(y) <= COURT_HALF_Y_M)
    alley = region & (np.abs(x) > HALF_SINGLES_WIDTH)
    order = np.argsort(track_ids, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(track_ids))]
    evidence = []
    ranked: dict[int, list[tuple[float, int]]] = {-1: [], 1: []}
    for track, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        observed = order[start:stop]
        court = observed[region[observed]]
        negative = court[y[court] < 0]
        positive = court[y[court] > 0]
        side = int(np.sign(len(positive) - len(negative)))
        on_side = negative if side == -1 else positive if side == 1 else court[:0]
        motion, pairs = _court_motion(frames[on_side], torso_m[on_side, :2], lag)
        score = len(on_side) * (
            1.0 + MAX_MOTION_BONUS * min(1.0, motion / MOTION_AT_FULL_BONUS_M)
        )
        evidence.append(
            (
                len(observed),
                len(court),
                (len(negative), len(positive)),
                int(alley[observed].sum()),
                side,
                motion,
                pairs,
                score,
            )
        )
        if (
            len(court) >= minimum_frames
            and side
            and (player_candidates is None or track in player_candidates)
        ):
            ranked[side].append((-score, track))
    selected = {
        track
        for group in ranked.values()
        for _, track in sorted(group)[:players_per_side]
    }
    decisions = []
    for track, row in enumerate(evidence):
        reason: SelectionReason
        if track in selected:
            reason = "selected"
        elif row[1] < minimum_frames:
            reason = "insufficient_residence"
        elif row[4] == 0:
            reason = "no_unique_side"
        else:
            reason = "side_capacity"
        decisions.append(PlayerDecision(*row, reason))
    return PlayerSelection(players_per_side, minimum_frames, tuple(decisions))
