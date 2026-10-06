from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import POINT_KIND_CODES

from ..contracts import ClipInput, Measurements


def compute(data: ClipInput, scope: NDArray[np.bool_]) -> Measurements:
    s, n = data.states, int(scope.sum())
    result = Measurements(counts={'frames': n, 'instances': int(s.count[scope].sum())})
    masks = dict(reviewed=s.reviewed, unreviewed=~s.reviewed, target=s.target, reference=~s.target,
                 reviewed_empty=s.reviewed & (s.count == 0), multi_ball=s.count > 1,
                 located=s.located > 0, evidence=s.evidence, supervised=s.supervision,
                 occluded=s.occluded > 0, boundary=data.breaks,
                 event_known=s.events != 3, event_hit=s.events == 1, event_bounce=s.events == 2)
    for name, mask in masks.items():
        count = int((mask & scope).sum())
        result.counts[f'annotation/{name}'] = count
        result.rates[f'annotation/{name}'] = (count, n)
    for name, code in POINT_KIND_CODES.items():
        result.counts[f'instances/{name}'] = int(s.kinds[scope, code].sum())
        result.rates[f'instances/{name}'] = (result.counts[f'instances/{name}'], result.counts['instances'])
        result.rates[f'frames/{name}'] = (int(((s.kinds[:, code] > 0) & scope).sum()), n)
    result.rates['annotation/occluded_located'] = (int((scope & (s.occluded_located > 0)).sum()), n)
    result.rates['annotation/occluded_unlocated'] = (int((scope & (s.occluded_unlocated > 0)).sum()), n)
    result.sample('annotation/frame_duration', data.durations[scope], 'seconds')
    result.sample('annotation/clip_duration', [data.durations[scope].sum()], 'seconds')
    result.views['annotation/seconds_total'] = float(data.durations[scope].sum())
    result.counts['annotation/notes_frames'] = sum(bool(scope[i]) for i in data.details.notes)
    result.rates['annotation/notes_frames'] = (result.counts['annotation/notes_frames'], n if data.details.notes_supported else 0)
    result.rates['annotation/original_available'] = (n if data.details.availability == 'available' else 0, n)
    return result
