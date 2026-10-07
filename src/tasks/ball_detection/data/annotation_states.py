"""One frame-aligned interpretation of stored annotations for review and statistics."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .store import POINT_KIND_CODES, POINT_KIND_NAMES, BallFrameStore, ClipRecord


@dataclass(frozen=True)
class AnnotationStates:
    reviewed: NDArray[np.bool_]
    target: NDArray[np.bool_]
    count: NDArray[np.int64]
    kinds: NDArray[np.int64]  # frames x point kinds: instance counts
    located: NDArray[np.int64]
    occluded: NDArray[np.int64]
    occluded_located: NDArray[np.int64]
    occluded_unlocated: NDArray[np.int64]
    xy: NDArray[np.float64]  # single instance only; ambiguous frames stay NaN
    track: NDArray[np.int64]
    times: NDArray[np.float64]
    boundaries: NDArray[np.bool_]
    events: NDArray[np.uint8]

    @property
    def visible_observation(self) -> NDArray[np.bool_]:
        return self.reviewed & (self.count == 1) & (self.kinds[:, POINT_KIND_CODES['observed']] == 1)

    @property
    def evidence(self) -> NDArray[np.bool_]:
        return self.reviewed & self.target & (self.count == 1) & (self.kinds[:, POINT_KIND_CODES['out_of_frame']] == 0)

    @property
    def supervision(self) -> NDArray[np.bool_]:
        return self.visible_observation & self.target

    def review_records(self) -> list[dict[str, Any]]:
        records = []
        evidence = self.evidence
        for i in range(len(self.count)):
            kinds = [POINT_KIND_NAMES[k] for k in range(len(POINT_KIND_NAMES)) for _ in range(int(self.kinds[i, k]))]
            reasons = []
            if not self.target[i]:
                reasons.append('reference_only')
            if not self.reviewed[i]:
                reasons.append('unreviewed')
            if not self.count[i]:
                reasons.append('no_ball')
            elif self.count[i] > 1:
                reasons.append('multiple_balls')
            elif kinds == ['out_of_frame']:
                reasons.append('out_of_frame')
            records.append(dict(kinds=kinds, located_count=int(self.located[i]), reviewed=bool(self.reviewed[i]),
                                is_target=bool(self.target[i]), evidence=bool(evidence[i]), exclusion_reasons=reasons))
        return records


def annotation_states(store: BallFrameStore, clip: ClipRecord) -> AnnotationStates:
    rows = store.clip_rows(clip)
    n = clip.frame_count
    counts = store.frames['inst_count'][rows].astype(np.int64)
    kinds = np.zeros((n, len(POINT_KIND_CODES)), np.int64)
    located, occluded = np.zeros(n, np.int64), np.zeros(n, np.int64)
    occluded_located, occluded_unlocated = np.zeros(n, np.int64), np.zeros(n, np.int64)
    xy = np.full((n, 2), np.nan, np.float64)
    track = np.full(n, -1, np.int64)
    start = int(store.frames['inst_start'][rows[0]])
    stop = start + int(counts.sum())
    owner: NDArray[np.int64] = np.repeat(np.arange(n), counts)
    coordinates = store.instances['xy'][start:stop]
    np.add.at(kinds, (owner, store.instances['point_kind'][start:stop]), 1)
    np.add.at(located, owner, np.isfinite(coordinates).all(axis=1))
    np.add.at(occluded, owner, store.instances['occluded'][start:stop])
    has_xy = np.isfinite(coordinates).all(axis=1)
    np.add.at(occluded_located, owner, store.instances['occluded'][start:stop] & has_xy)
    np.add.at(occluded_unlocated, owner, store.instances['occluded'][start:stop] & ~has_xy)
    single = counts == 1
    offsets = store.frames['inst_start'][rows[single]]
    xy[single] = store.instances['xy'][offsets]
    track[single] = store.instances['track_index'][offsets]
    pts = store.frames['pts'][rows]
    times = (pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))
    return AnnotationStates(store.frames['annotated'][rows].copy(), store.frames['is_target'][rows].copy(),
                            counts, kinds, located, occluded, occluded_located, occluded_unlocated, xy, track, times,
                            store.frames['segment_break'][rows].copy(), store.frames['event'][rows].copy())
