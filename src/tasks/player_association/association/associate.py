"""Cross-camera association of camera-local person tracks into rally players.

1. **Segments.** Every track is cut at its identity-switch candidates
   (``geometry.switches``). A segment with fewer than ``min_segment_s`` of
   valid footpoints is excluded (``too_few_footpoints``): it carries too little
   evidence to be placed.
2. **Scores.** For every pair of remaining segments:
   different cameras: geometry log-likelihood ratio (``geometry.affinity``)
   plus, when configured, the appearance log-likelihood ratio
   (``appearance.affinity``); same camera: forbidden when both observe more
   than ``max_handoff_overlap_s`` of common frames (one camera cannot see one
   person twice; a short overlap is a tracker handing a person over to a new
   track), ``continuity_score`` between consecutive segments of one track,
   otherwise 0.
3. **Clustering.** ``cluster_multiview`` finds the maximum-score partition
   into identities (every segment is its own view; the same-camera rule is
   the ``allowed`` mask).
4. **Players.** An identity's position is the per-frame median of its
   segments' footpoints; its presence is the number of frames that position is
   in the play region, and its side of the net is the sign of the median ``y``
   there. On each side the ``players_per_side`` most present identities are
   the players; the others are excluded (``not_selected``).
5. **Stops.** The association raises :class:`AssociationUndecided`, never
   returns a guess, when a side lacks a player present for
   ``min_presence_fraction`` of the clip, when the next identity on a side is
   within ``max_runner_up_ratio`` of a player's presence, when a decision that
   touches a player segment has a margin below ``min_margin`` (the best and
   the next-best clustering are nearly tied) and both of its segments last
   ``max_undecided_segment_s`` or longer, or when there are more segments than
   the solver accepts or it cannot prove an optimum within ``time_limit_s``. A nearly tied decision whose shorter segment is briefer
   than that does not stop the clip: that segment is excluded
   (``undecided``), whatever identity the best clustering gave it.
6. **Output.** Every frame of a player segment carries the player ID. Where two
   segments of one camera and player both observe a frame (a handoff), the
   segment with more footpoints keeps the ID and the other gets ``-1`` there
   (``handoff_frames`` in the diagnostics).

Player IDs are ``0..`` in side order (``y < 0`` first), then by presence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.appearance.affinity import (
    AppearanceAffinityConfig,
    appearance_score,
    segment_embedding,
)
from src.tasks.player_association.geometry.affinity import (
    GeometryAffinityConfig,
    geometry_score,
)
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_distance,
    ground_footpoints,
)
from src.tasks.player_association.geometry.region import (
    PlayRegionConfig,
    in_play_region,
)
from src.tasks.player_association.geometry.switches import (
    SwitchConfig,
    switch_candidates,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.matching import SolverTimeLimit, cluster_multiview, decision_margins
from src.utils.matching.multiview_clustering import MAX_ITEMS

TOO_MANY_SEGMENTS = "too_many_segments"
PLAYER_NOT_FOUND = "player_not_found"
AMBIGUOUS_PLAYERS = "ambiguous_players"
AMBIGUOUS_ASSOCIATION = "ambiguous_association"
SOLVER_TIME_LIMIT = "solver_time_limit"


class AssociationUndecided(RuntimeError):
    """The association stopped; ``diagnostics`` holds every score and decision made up to the stop."""

    def __init__(self, reason: str, message: str, diagnostics: dict[str, Any]) -> None:
        self.reason = reason
        self.diagnostics = diagnostics
        super().__init__(message)


@dataclass(frozen=True)
class AssociationConfig:
    players_per_side: int  # 1 for singles, 2 for doubles
    footpoints: FootpointConfig
    switches: SwitchConfig
    geometry: GeometryAffinityConfig
    appearance: AppearanceAffinityConfig | None  # None: geometry only
    continuity_score: float
    min_segment_s: float
    max_handoff_overlap_s: float
    region: PlayRegionConfig
    min_presence_fraction: float
    max_runner_up_ratio: float
    min_margin: float
    max_undecided_segment_s: float
    time_limit_s: float = 10.

    def __post_init__(self) -> None:
        if self.players_per_side not in (1, 2):
            raise ValueError("players_per_side must be 1 (singles) or 2 (doubles)")
        if not (self.continuity_score >= 0 and self.min_segment_s > 0 and self.max_handoff_overlap_s >= 0 and 0 < self.min_presence_fraction <= 1
                and 0 < self.max_runner_up_ratio <= 1 and self.min_margin >= 0 and self.max_undecided_segment_s >= 0 and self.time_limit_s > 0):
            raise ValueError(f"Invalid association config: {self}")


@dataclass(frozen=True)
class TrackAppearance:
    """Embeddings of the sampled crops of one track."""

    frames: NDArray[np.int64]  # (K,)
    embeddings: NDArray[np.float32]  # (K, E)


@dataclass(frozen=True)
class CameraTracks:
    """Camera-local tracks with the side-resolved camera of their view."""

    camera: PinholeCamera
    image_size: tuple[int, int]  # width, height
    track_ids: NDArray[np.int64]  # (D,)
    boxes_xyxy: NDArray[np.floating]  # (D, T, 4)
    observed: NDArray[np.bool_]  # (D, T)
    appearance: tuple[TrackAppearance, ...] | None = None  # one per track

    def __post_init__(self) -> None:
        tracks, frames = self.observed.shape
        if self.track_ids.shape != (tracks,) or self.boxes_xyxy.shape != (tracks, frames, 4) or self.observed.dtype != np.bool_:
            raise ValueError(f"{self.camera.camera_id}: tracks must be (D,), (D, T, 4), boolean (D, T)")
        if len(np.unique(self.track_ids)) != tracks:
            raise ValueError(f"{self.camera.camera_id}: track IDs must be unique")
        if self.appearance is not None and len(self.appearance) != tracks:
            raise ValueError(f"{self.camera.camera_id}: one appearance per track is required")


@dataclass(frozen=True)
class Segment:
    camera: int
    row: int
    track_id: int
    start: int
    end: int  # exclusive
    observed_frames: int
    footpoint_frames: int


@dataclass(frozen=True)
class Association:
    camera_ids: tuple[str, ...]
    track_ids: tuple[NDArray[np.int64], ...]
    player_ids: tuple[NDArray[np.int64], ...]  # per camera (D, T); -1 = not a player
    diagnostics: dict[str, Any]


def _segments(cameras: Sequence[CameraTracks], points: list[NDArray[np.float64]], valid: list[NDArray[np.bool_]],
              fps: float, config: SwitchConfig) -> list[Segment]:
    segments = []
    for view, tracks in enumerate(cameras):
        frames = tracks.observed.shape[1]
        for row, track_id in enumerate(tracks.track_ids.tolist()):
            cuts = [0, *switch_candidates(points[view][row], valid[view][row], fps, config), frames]
            for start, end in zip(cuts, cuts[1:], strict=False):
                segments.append(Segment(view, row, int(track_id), start, end, int(tracks.observed[row, start:end].sum()),
                                        int(valid[view][row, start:end].sum())))
    return segments


def _window(mask: NDArray[np.bool_], segment: Segment) -> NDArray[np.bool_]:
    windowed = np.zeros_like(mask)
    windowed[segment.start:segment.end] = mask[segment.start:segment.end]
    return windowed


def _segment_json(segment: Segment, camera_ids: tuple[str, ...]) -> dict[str, Any]:
    return {"camera": camera_ids[segment.camera], "track_id": segment.track_id, "start": segment.start, "end": segment.end,
            "observed_frames": segment.observed_frames, "footpoint_frames": segment.footpoint_frames}


def associate(cameras: Sequence[CameraTracks], fps: float, config: AssociationConfig) -> Association:
    """Players of a clip from camera-local tracks; see the module docstring."""
    camera_ids = tuple(tracks.camera.camera_id for tracks in cameras)
    if len(cameras) < 2 or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("Association needs two or more cameras with unique IDs")
    frames = cameras[0].observed.shape[1]
    if any(tracks.observed.shape[1] != frames for tracks in cameras) or not frames or fps <= 0:
        raise ValueError("Cameras must share a nonempty timeline and fps must be positive")
    if config.appearance is not None and any(tracks.appearance is None for tracks in cameras):
        raise ValueError("Appearance scoring is configured but a camera has no track appearance")
    footpoints = [ground_footpoints(tracks.boxes_xyxy, tracks.observed, tracks.camera, tracks.image_size[1], config.footpoints)
                  for tracks in cameras]
    points, valid = [p for p, _ in footpoints], [v for _, v in footpoints]
    segments = _segments(cameras, points, valid, fps, config.switches)
    min_frames = config.min_segment_s * fps
    candidates = [index for index, segment in enumerate(segments) if segment.footpoint_frames >= min_frames]
    diagnostics: dict[str, Any] = {"frames": frames, "fps": fps,
                                   "segments": [{**_segment_json(s, camera_ids), "candidate": i in candidates} for i, s in enumerate(segments)]}
    count = len(candidates)
    if count > MAX_ITEMS:
        raise AssociationUndecided(TOO_MANY_SEGMENTS, f"{count} segments exceed the solver limit {MAX_ITEMS}", diagnostics)

    items = [segments[i] for i in candidates]
    item_points = [np.where(_window(valid[s.camera][s.row], s)[:, None], points[s.camera][s.row], 0.) for s in items]
    item_valid = [_window(valid[s.camera][s.row], s) for s in items]
    item_observed = [_window(cameras[s.camera].observed[s.row], s) for s in items]
    embeddings: list[NDArray[np.float64] | None] = [None] * count
    if config.appearance is not None:
        for k, segment in enumerate(items):
            appearance = cameras[segment.camera].appearance
            assert appearance is not None
            sampled = appearance[segment.row]
            embeddings[k] = segment_embedding(sampled.frames, sampled.embeddings, segment.start, segment.end)
    scores = np.zeros((count, count))
    allowed: NDArray[np.bool_] = np.ones((count, count), bool)
    evidence: list[dict[str, Any]] = []
    for a in range(count):
        for b in range(a + 1, count):
            first, second = items[a], items[b]
            record: dict[str, Any] = {"a": a, "b": b}
            if first.camera == second.camera:
                overlap = int((item_observed[a] & item_observed[b]).sum())
                if overlap > config.max_handoff_overlap_s * fps:
                    allowed[a, b] = allowed[b, a] = False
                    record["forbidden"] = f"same_camera_{overlap}_shared_frames"
                elif first.row == second.row and first.end == second.start:
                    scores[a, b] = scores[b, a] = config.continuity_score
                    record["continuity"] = config.continuity_score
            else:
                distance = ground_distance(item_points[a], item_valid[a], item_points[b], item_valid[b])
                score = geometry_score(distance, fps, config.geometry)
                record.update(shared_frames=distance.shared_frames, median_m=None if distance.shared_frames == 0 else distance.median_m,
                              geometry=score)
                if config.appearance is not None and embeddings[a] is not None and embeddings[b] is not None:
                    cosine = float(np.clip(embeddings[a] @ embeddings[b], -1., 1.))  # type: ignore[operator]
                    record.update(cosine=cosine, appearance=appearance_score(cosine, config.appearance))
                    score += record["appearance"]
                scores[a, b] = scores[b, a] = score
            if len(record) > 2:
                evidence.append(record)
    diagnostics["candidates"] = candidates
    diagnostics["pairs"] = evidence
    try:
        clustering = cluster_multiview(scores, np.arange(count), allowed=allowed, time_limit_s=config.time_limit_s)
    except SolverTimeLimit as limited:
        raise AssociationUndecided(SOLVER_TIME_LIMIT, str(limited), diagnostics) from limited
    labels = clustering.labels
    diagnostics["clustering"] = {"objective": clustering.objective, "identity_of_candidate": labels.tolist()}

    identities = []
    for identity in range(int(labels.max()) + 1 if count else 0):
        members = np.flatnonzero(labels == identity)
        stacked = np.stack([np.where(item_valid[k][:, None], item_points[k], np.nan) for k in members])
        seen = item_valid[members[0]].copy()
        for k in members[1:]:
            seen |= item_valid[k]
        position = np.full((frames, 2), np.nan)
        position[seen] = np.nanmedian(stacked[:, seen], axis=0)
        inside = seen & in_play_region(np.nan_to_num(position), config.region)
        half = 0 if not inside.any() else (-1 if np.median(position[inside, 1]) < 0 else 1)
        identities.append({"identity": identity, "candidates": members.tolist(), "presence_frames": int(inside.sum()),
                           "seen_frames": int(seen.sum()), "side": half})
    diagnostics["identities"] = identities
    required = config.min_presence_fraction * frames
    players: list[int] = []
    selection: dict[str, Any] = {}
    for half in (-1, 1):
        ranked = sorted((item for item in identities if item["side"] == half), key=lambda item: (-item["presence_frames"], item["identity"]))
        chosen = ranked[:config.players_per_side]
        runner_up = ranked[config.players_per_side] if len(ranked) > config.players_per_side else None
        selection[str(half)] = {"players": [item["identity"] for item in chosen],
                                "runner_up": None if runner_up is None else runner_up["identity"]}
        diagnostics["selection"] = selection
        if len(chosen) < config.players_per_side or chosen[-1]["presence_frames"] < required:
            raise AssociationUndecided(PLAYER_NOT_FOUND, f"Side y{'<' if half < 0 else '>'}0 lacks {config.players_per_side} identities "
                                       f"present for {required:.0f} frames", diagnostics)
        if runner_up is not None and runner_up["presence_frames"] > config.max_runner_up_ratio * chosen[-1]["presence_frames"]:
            raise AssociationUndecided(AMBIGUOUS_PLAYERS, f"Identity {runner_up['identity']} ({runner_up['presence_frames']} frames) is too "
                                       f"close to player identity {chosen[-1]['identity']} ({chosen[-1]['presence_frames']} frames)", diagnostics)
        players += [item["identity"] for item in chosen]

    relevant = np.isin(labels, players)
    try:
        clustering = decision_margins(clustering, scores, np.arange(count), allowed=allowed, items=relevant, time_limit_s=config.time_limit_s)
    except SolverTimeLimit as limited:
        raise AssociationUndecided(SOLVER_TIME_LIMIT, str(limited), diagnostics) from limited
    assert clustering.margins is not None
    ambiguous = clustering.ambiguous_pairs(config.min_margin, items=relevant)
    finite = clustering.margins[np.isfinite(clustering.margins) & (relevant[:, None] | relevant[None, :])]
    diagnostics["min_player_margin"] = float(finite.min()) if len(finite) else None
    diagnostics["ambiguous_pairs"] = [{"a": a, "b": b, "margin": margin, "linked": bool(clustering.linked[a, b])} for a, b, margin in ambiguous]
    undecided: set[int] = set()
    decisive = [(a, b, margin) for a, b, margin in ambiguous
                if min(items[a].footpoint_frames, items[b].footpoint_frames) >= config.max_undecided_segment_s * fps]
    if decisive:
        a, b, margin = decisive[0]
        raise AssociationUndecided(AMBIGUOUS_ASSOCIATION, f"{len(decisive)} player decisions between long segments have margins below "
                                   f"{config.min_margin}; the smallest is {margin:.3f} for candidates {a} and {b}", diagnostics)
    for a, b, _ in ambiguous:
        undecided.add(min((a, b), key=lambda k: (items[k].footpoint_frames, k)))
    diagnostics["undecided_candidates"] = sorted(undecided)

    player_of_identity = {identity: player for player, identity in enumerate(players)}
    player_ids = [np.full(tracks.observed.shape, -1, np.int64) for tracks in cameras]
    handoffs = []
    for k in sorted(range(count), key=lambda k: (-items[k].footpoint_frames, k)):
        segment, player = items[k], player_of_identity.get(int(labels[k]), -1)
        if player < 0 or k in undecided:
            continue
        ids, observed = player_ids[segment.camera], cameras[segment.camera].observed
        span = slice(segment.start, segment.end)
        taken = ((ids[:, span] == player) & observed[:, span]).any(0) & observed[segment.row, span]
        ids[segment.row, span] = np.where(taken, -1, player)
        if taken.any():
            handoffs.append({"candidate": k, "player_id": player, "frames": int(taken.sum())})
    diagnostics["handoff_frames"] = handoffs
    for view, ids in enumerate(player_ids):
        visible = np.where(cameras[view].observed, ids, -1)
        for player in range(len(players)):
            if ((visible == player).sum(0) > 1).any():
                raise RuntimeError(f"{camera_ids[view]} observes player {player} through two tracks in one frame")
    diagnostics["players"] = [{"player_id": player, "identity": identity} for identity, player in player_of_identity.items()]
    return Association(camera_ids, tuple(tracks.track_ids for tracks in cameras), tuple(player_ids), diagnostics)
