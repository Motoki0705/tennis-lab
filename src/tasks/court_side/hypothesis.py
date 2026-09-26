"""Court side decision: a geometric hypothesis test over multi-view ball observations.

Each camera is calibrated in its own camera-local court, which is symmetric
under the half-turn ``Rz(pi)``; one camera alone cannot tell which end it
faces. With the reference camera fixed, every other camera is either aligned
with the reference court or half-turned, giving ``2^(V-1)`` hypotheses. The
single ball needs no cross-camera identity matching: each hypothesis
triangulates the ball in every frame seen by two or more views and is scored
by reprojection error and physical plausibility. The best hypothesis must be
absolutely consistent and beat the runner-up by a margin; otherwise the
decision stops with :class:`CourtSideUndecided`, carrying every hypothesis'
score. There is no fallback to another decision method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.multiview_consistency import (
    ConsistencyBounds,
    score_multiview_points,
)
from src.utils.geometry.triangulation import PinholeCamera

# Stop reasons of CourtSideUndecided.
INSUFFICIENT_FRAMES = "insufficient_frames"
DISCONNECTED_VIEWS = "disconnected_views"
NO_CONSISTENT_HYPOTHESIS = "no_consistent_hypothesis"
AMBIGUOUS_MARGIN = "ambiguous_margin"


@dataclass(frozen=True)
class CourtSideConfig:
    """Decision thresholds.

    ``reprojection_px`` is in the pixels of the supplied observations (the
    caller scales resolution-dependent thresholds). ``min_frames`` counts
    frames with two or more observing views, overall and for the pair graph
    that must connect every camera to the reference.
    """

    reprojection_px: float
    min_frames: int
    max_cost: float
    min_support: float
    min_margin: float
    height_range_m: tuple[float, float] = (-0.2, 20.0)
    max_abs_xy_m: float = 40.0
    min_ray_angle_deg: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.reprojection_px) or self.reprojection_px <= 0 or self.min_frames < 1:
            raise ValueError("Court side needs a positive reprojection threshold and frame count")
        for value in (self.max_cost, self.min_support, self.min_margin):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Court side cost/support/margin thresholds must be in [0,1]")
        ConsistencyBounds(self.height_range_m, self.max_abs_xy_m, self.min_ray_angle_deg)

    @property
    def bounds(self) -> ConsistencyBounds:
        return ConsistencyBounds(self.height_range_m, self.max_abs_xy_m, self.min_ray_angle_deg)


@dataclass(frozen=True)
class HypothesisScore:
    """One half-turn assignment scored on ``frames`` multi-view frames.

    ``cost`` is the mean per-frame normalized reprojection cost in ``[0,1]``;
    ``support`` the fraction of frames that are plausible with every view
    within the reprojection threshold.
    """

    view_half_turns: tuple[bool, ...]
    cost: float
    support: float
    frames: int


@dataclass(frozen=True)
class CourtSideDecision:
    """``hypotheses`` are sorted best first; ``margin`` is runner-up cost minus best cost."""

    camera_ids: tuple[str, ...]
    reference_camera: str
    view_half_turns: tuple[bool, ...]
    hypotheses: tuple[HypothesisScore, ...]
    margin: float
    frames: int


class CourtSideUndecided(RuntimeError):
    """The ball evidence does not determine the sides; ``reason`` is one of the stop reasons."""

    def __init__(self, reason: str, message: str, *, hypotheses: tuple[HypothesisScore, ...] = (),
                 frames: int = 0, pair_frames: dict[str, int] | None = None) -> None:
        self.reason = reason
        self.hypotheses = hypotheses
        self.frames = frames
        self.pair_frames = {} if pair_frames is None else pair_frames
        summary = "; ".join(f"{list(h.view_half_turns)} cost={h.cost:.3f} support={h.support:.3f}" for h in hypotheses)
        super().__init__(f"{reason}: {message}" + (f" [{summary}]" if summary else ""))


def half_turn_hypotheses(views: int, reference: int) -> tuple[tuple[bool, ...], ...]:
    """Every assignment with the reference camera unturned, in lexicographic order."""
    if views < 2 or not 0 <= reference < views:
        raise ValueError("Side hypotheses need two or more views and a reference among them")
    choices = [(False,) if view == reference else (False, True) for view in range(views)]
    return tuple(tuple(turns) for turns in product(*choices))


def pair_frame_counts(visible: NDArray[np.bool_]) -> NDArray[np.int64]:
    """``(V,V)`` number of frames both views observe."""
    counts = np.einsum("at,bt->ab", visible.astype(np.int64), visible.astype(np.int64))
    np.fill_diagonal(counts, 0)
    return counts


def score_hypothesis(cameras: tuple[PinholeCamera, ...], view_half_turns: tuple[bool, ...],
                     uv_px: NDArray[np.floating], visible: NDArray[np.bool_], config: CourtSideConfig) -> HypothesisScore:
    """Score one assignment on the frames with two or more observing views."""
    chosen = visible.sum(0) >= 2
    turned = tuple(camera.half_turned(turn) for camera, turn in zip(cameras, view_half_turns, strict=True))
    if not chosen.any():
        raise ValueError("Scoring needs at least one frame observed by two views")
    score = score_multiview_points(uv_px[:, chosen], visible[:, chosen], turned, threshold_px=config.reprojection_px, bounds=config.bounds)
    return HypothesisScore(tuple(view_half_turns), float(score.cost.mean()), float(score.support.mean()), int(chosen.sum()))


@dataclass(frozen=True)
class BallSideEvidence:
    """Everything the judgement needs, independent of the decision thresholds.

    ``pair_frames`` ``(V,V)`` counts frames both views observe; ``hypotheses``
    are sorted best first and empty when no frame has two observing views.
    """

    camera_ids: tuple[str, ...]
    reference_camera: str
    frames: int
    pair_frames: NDArray[np.int64]
    hypotheses: tuple[HypothesisScore, ...]

    def pair_record(self) -> dict[str, int]:
        ids = self.camera_ids
        return {f"{ids[a]}-{ids[b]}": int(self.pair_frames[a, b]) for a, b in combinations(range(len(ids)), 2)}


def collect_side_evidence(cameras: tuple[PinholeCamera, ...], reference_camera: str, uv_px: NDArray[np.floating],
                          visible: NDArray[np.bool_], config: CourtSideConfig) -> BallSideEvidence:
    """Score every hypothesis on the ball stream (``(V,T,2)`` pixels, ``(V,T)`` visibility).

    Only ``reprojection_px`` and the plausibility bounds of ``config`` are used.
    """
    ids = tuple(camera.camera_id for camera in cameras)
    if len(set(ids)) != len(ids) or reference_camera not in ids:
        raise ValueError("Court side needs unique camera IDs including the reference")
    if uv_px.ndim != 3 or uv_px.shape[0] != len(ids) or uv_px.shape[-1] != 2 or visible.shape != uv_px.shape[:-1] or visible.dtype != np.bool_:
        raise ValueError("Ball observations must be (V,T,2) pixels with (V,T) boolean visibility")
    frames = int((visible.sum(0) >= 2).sum())
    hypotheses: tuple[HypothesisScore, ...] = ()
    if frames:
        hypotheses = tuple(sorted((score_hypothesis(cameras, turns, uv_px, visible, config)
                                   for turns in half_turn_hypotheses(len(ids), ids.index(reference_camera))),
                                  key=lambda h: (h.cost, h.view_half_turns)))
    return BallSideEvidence(ids, reference_camera, frames, pair_frame_counts(visible), hypotheses)


def judge_side_evidence(evidence: BallSideEvidence, config: CourtSideConfig) -> CourtSideDecision:
    """Apply the decision thresholds, or raise :class:`CourtSideUndecided`."""
    ids, frames, hypotheses = evidence.camera_ids, evidence.frames, evidence.hypotheses
    pairs = evidence.pair_record()
    if frames < config.min_frames:
        raise CourtSideUndecided(INSUFFICIENT_FRAMES, f"{frames} multi-view ball frames; {config.min_frames} required",
                                 hypotheses=hypotheses, frames=frames, pair_frames=pairs)
    connected = {ids.index(evidence.reference_camera)}
    graph = evidence.pair_frames >= config.min_frames
    while True:
        expanded = connected | {view for old in connected for view in np.flatnonzero(graph[old]).tolist()}
        if expanded == connected:
            break
        connected = expanded
    if len(connected) != len(ids):
        missing = [ids[view] for view in range(len(ids)) if view not in connected]
        raise CourtSideUndecided(DISCONNECTED_VIEWS, f"Cameras {missing} share too few ball frames with the reference's component",
                                 hypotheses=hypotheses, frames=frames, pair_frames=pairs)
    best = hypotheses[0]
    margin = hypotheses[1].cost - best.cost
    if best.cost > config.max_cost or best.support < config.min_support:
        raise CourtSideUndecided(NO_CONSISTENT_HYPOTHESIS, "The best hypothesis lacks absolute geometric support",
                                 hypotheses=hypotheses, frames=frames, pair_frames=pairs)
    if margin < config.min_margin:
        raise CourtSideUndecided(AMBIGUOUS_MARGIN, f"Runner-up is only {margin:.3f} worse than the best",
                                 hypotheses=hypotheses, frames=frames, pair_frames=pairs)
    return CourtSideDecision(ids, evidence.reference_camera, best.view_half_turns, hypotheses, margin, frames)


def decide_court_side(cameras: tuple[PinholeCamera, ...], reference_camera: str, uv_px: NDArray[np.floating],
                      visible: NDArray[np.bool_], config: CourtSideConfig) -> CourtSideDecision:
    """Decide every camera's half-turn from one ball stream, or raise :class:`CourtSideUndecided`.

    ``cameras`` are camera-local calibrations; ``uv_px`` ``(V,T,2)`` and
    ``visible`` ``(V,T)`` hold at most one ball per camera and frame.
    """
    return judge_side_evidence(collect_side_evidence(cameras, reference_camera, uv_px, visible, config), config)
