"""Dev-only fragment linking and exclusive-court dwell, with no score/GT gate.

The calibrated z=0 plane uses the court model's singles width as a dwell
core, doubles width as the outer acceptance corridor and 5 m of baseline
runoff. Neither region expands laterally past the target court's sidelines.
The inner core adds 1.37 m of protection against noisy box-bottom projections
at the lateral boundary; a track parked only in that border cannot qualify.
These are conservative singles-dev rules, not a general venue segmentation.
Calibration/footpoint error can still misplace people; no such error bound is
claimed. Outside-corridor observations are excluded even on a selected track.

Split real observations at >0.5 s gaps or impossible jumps, then connect
temporally neighbouring fragments by court position. A <=0.2 s overlapping
handoff also needs spatial agreement. Available CLIP embeddings veto a link
below cosine .8; missing appearance is explicitly recorded. Mutual nearest
links must beat each runner-up by .2 in spatial-distance/limit + gap/max-gap.
Ambiguous fragments
shorter than 1 s are excluded, never forced into a chain. Count distinct
observed core frames per chain; require 25% of the clip, THEN cap at 6 chains.
No interpolated frame counts as dwell, and overlapping handoffs count once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.appearance.affinity import segment_embedding
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH, HALF_SINGLES_WIDTH


@dataclass(frozen=True)
class LinkingConfig:
    max_gap_s: float = .5
    max_handoff_s: float = .2
    position_slack_m: float = .8
    max_speed_m_s: float = 6.
    min_cosine: float = .8
    ambiguity_margin: float = .2
    ambiguous_min_s: float = 1.
    baseline_margin_m: float = 5.
    min_presence_fraction: float = .25
    max_candidates: int = 6

    def __post_init__(self) -> None:
        values = (self.max_gap_s, self.max_handoff_s, self.position_slack_m, self.max_speed_m_s,
                  self.min_cosine, self.ambiguity_margin, self.ambiguous_min_s,
                  self.baseline_margin_m, self.min_presence_fraction)
        if not np.isfinite(values).all() or min(values) < 0 or self.max_gap_s <= 0 \
                or self.position_slack_m <= 0 or not 0 < self.min_presence_fraction <= 1 \
                or not 0 <= self.min_cosine <= 1 or self.max_candidates < 1:
            raise ValueError('Invalid fragment linking configuration')


def exclusive_region(points: NDArray[np.floating], config: LinkingConfig, *, core: bool) -> NDArray[np.bool_]:
    if points.shape[-1:] != (2,):
        raise ValueError('Court coordinates must be (..., 2)')
    half_width = HALF_SINGLES_WIDTH if core else HALF_DOUBLES_WIDTH
    return (np.abs(points[..., 0]) <= half_width) & (np.abs(points[..., 1]) <= HALF_LENGTH + config.baseline_margin_m)


@dataclass(frozen=True)
class Fragment:
    row: int
    frames: NDArray[np.int64]
    points: NDArray[np.float64]
    embedding: NDArray[np.float64] | None


def _fragments(tracks: CameraTracks, points: NDArray[np.float64], valid: NDArray[np.bool_],
               fps: float, config: LinkingConfig) -> list[Fragment]:
    fragments = []
    for row in range(len(tracks.track_ids)):
        frames = np.flatnonzero(valid[row]).astype(np.int64)
        gaps = np.diff(frames) / fps
        distances = np.linalg.norm(np.diff(points[row, frames], axis=0), axis=1)
        cuts = np.flatnonzero((gaps > config.max_gap_s) | (distances > config.position_slack_m + config.max_speed_m_s * gaps)) + 1
        for indices in np.split(frames, cuts):
            if not len(indices):
                continue
            embedding = None
            if tracks.appearance is not None:
                a = tracks.appearance[row]
                embedding = segment_embedding(a.frames, a.embeddings, int(indices[0]), int(indices[-1]) + 1)
            fragments.append(Fragment(row, indices, points[row, indices], embedding))
    return fragments


def _pair(a: Fragment, b: Fragment, fps: float, config: LinkingConfig) -> dict[str, Any] | None:
    if a.frames[0] >= b.frames[0] or a.frames[-1] >= b.frames[-1]:
        return None
    gap = (int(b.frames[0]) - int(a.frames[-1])) / fps
    if gap > config.max_gap_s or gap < -config.max_handoff_s:
        return None
    shared, ia, ib = np.intersect1d(a.frames, b.frames, return_indices=True)
    if len(shared):
        distance = float(np.median(np.linalg.norm(a.points[ia] - b.points[ib], axis=1)))
    else:
        distance = float(np.linalg.norm(a.points[-1] - b.points[0]))
    limit = config.position_slack_m + config.max_speed_m_s * max(gap, 0.)
    if distance > limit:
        return None
    cosine = None
    if a.embedding is not None and b.embedding is not None:
        cosine = float(a.embedding @ b.embedding)
        if cosine < config.min_cosine:
            return None
    return {'distance_m': distance, 'gap_s': gap, 'overlap_frames': len(shared),
            'cosine': cosine, 'appearance': 'available' if cosine is not None else 'missing',
            'cost': distance / limit + max(gap, 0.) / config.max_gap_s}


def select_linked_candidates(tracks: CameraTracks, fps: float, config: LinkingConfig,
                             footpoints: FootpointConfig, *, link_fragments: bool = True) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    points, valid = ground_footpoints(tracks.boxes_xyxy, tracks.observed, tracks.camera, tracks.image_size[1], footpoints)
    # Excluding outside observations before splitting also prevents dwell from
    # leaking across a long excursion into another court on the same raw ID.
    outer = valid & exclusive_region(points, config, core=False)
    core = outer & exclusive_region(points, config, core=True)
    fragments = _fragments(tracks, points, outer, fps, config)
    pairs = []
    for i, a in enumerate(fragments):
        for j, b in enumerate(fragments):
            value = _pair(a, b, fps, config) if link_fragments else None
            if value is not None:
                pairs.append({'a': i, 'b': j, **value})
    outgoing: dict[int, list[dict[str, Any]]] = {}
    incoming: dict[int, list[dict[str, Any]]] = {}
    for pair in sorted(pairs, key=lambda p: (p['cost'], p['a'], p['b'])):
        outgoing.setdefault(pair['a'], []).append(pair)
        incoming.setdefault(pair['b'], []).append(pair)
    ambiguous: set[int] = set()
    for options in (*outgoing.values(), *incoming.values()):
        if len(options) > 1 and options[1]['cost'] - options[0]['cost'] < config.ambiguity_margin:
            for pair in options:
                if pair['cost'] - options[0]['cost'] < config.ambiguity_margin:
                    ambiguous.update((pair['a'], pair['b']))
    excluded = {i for i in ambiguous if len(fragments[i].frames) / fps < config.ambiguous_min_s}
    groups = [{i} for i in range(len(fragments))]
    owner = list(range(len(fragments)))
    links = []
    for pair in sorted(pairs, key=lambda p: (p['cost'], p['a'], p['b'])):
        a, b = pair['a'], pair['b']
        if a in ambiguous or b in ambiguous or outgoing[a][0] is not pair or incoming[b][0] is not pair:
            continue
        ga, gb = owner[a], owner[b]
        if ga == gb:
            continue
        # Transitive chains must also respect the same-camera exclusion rule.
        if any(len(np.intersect1d(fragments[x].frames, fragments[y].frames)) > config.max_handoff_s * fps
               for x in groups[ga] for y in groups[gb]):
            continue
        groups[ga] |= groups[gb]
        for member in groups[gb]:
            owner[member] = ga
        groups[gb] = set()
        links.append(pair)
    candidates: list[dict[str, Any]] = []
    for members in groups:
        usable = sorted(members - excluded)
        if not usable:
            continue
        frames = np.unique(np.concatenate([f.frames[core[f.row, f.frames]] for i in usable for f in [fragments[i]]]))
        candidates.append({'fragments': usable, 'in_core_frames': len(frames),
                           'track_ids': sorted({int(tracks.track_ids[fragments[i].row]) for i in usable})})
    required = tracks.observed.shape[1] * config.min_presence_fraction
    ranked = sorted((c for c in candidates if c['in_core_frames'] >= required),
                    key=lambda c: (-c['in_core_frames'], c['fragments'][0]))
    selected_groups = ranked[:config.max_candidates]
    selected = np.zeros_like(tracks.observed)
    for candidate in selected_groups:
        occupied = np.zeros(tracks.observed.shape[1], bool)
        # A handoff contributes one observed box per group/frame, with earlier
        # source ID as a deterministic tie-break, not the detector score.
        for i in sorted(candidate['fragments'], key=lambda i: (int(tracks.track_ids[fragments[i].row]), i)):
            f = fragments[i]
            keep = f.frames[~occupied[f.frames]]
            selected[f.row, keep] = True
            occupied[keep] = True
    for candidate in candidates:
        candidate['selected'] = candidate in selected_groups
        candidate['reason'] = ('selected' if candidate['selected'] else
            'candidate_cap' if candidate['in_core_frames'] >= required else 'insufficient_dwell')
    diagnostic = {'required_frames': required, 'all_tracks': len(tracks.track_ids),
        'fragments': [{'row': f.row, 'track_id': int(tracks.track_ids[f.row]), 'start': int(f.frames[0]),
                       'end': int(f.frames[-1]) + 1, 'observed_frames': len(f.frames),
                       'appearance': f.embedding is not None, 'ambiguous': i in ambiguous,
                       'excluded_short_ambiguous': i in excluded} for i, f in enumerate(fragments)],
        'links': links, 'groups': candidates, 'selected_groups': len(selected_groups),
        'outside_observations': int((tracks.observed & ~outer).sum()),
        'valid_footpoints': int(valid.sum()), 'within_corridor': int(outer.sum())}
    return selected, diagnostic
