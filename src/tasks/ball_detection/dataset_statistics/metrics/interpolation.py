from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..contracts import ClipInput, Measurements
from ..scopes import runs


def compute(data: ClipInput, scope: NDArray[np.bool_]) -> Measurements:
    result = Measurements()
    intervals = runs(scope & (data.states.kinds[:, 1] > 0), data.breaks)
    result.counts['interpolation/runs'] = len(intervals)
    result.sample('interpolation/run_frames', [b - a for a, b in intervals], 'frames')
    result.sample('interpolation/run_seconds', [data.durations[a:b].sum() for a, b in intervals], 'seconds')
    pairs = sorted({pair for frame, pair in data.details.endpoints.items() if scope[frame]})
    distances = [np.linalg.norm((data.states.xy[b] - data.states.xy[a]) / data.clip.scale) for a, b in pairs]
    result.sample('interpolation/endpoint_distance', distances, 'source_px')
    result.sample('interpolation/endpoint_seconds', [data.states.times[b] - data.states.times[a] for a, b in pairs], 'seconds')
    result.counts['interpolation/endpoint_pairs'] = len(pairs)
    result.rates['interpolation/endpoint_coverage'] = (sum(bool(scope[frame]) for frame in data.details.endpoints), int((scope & (data.states.kinds[:, 1] > 0)).sum()))
    result.views['interpolation/availability'] = data.details.availability
    result.views['interpolation/runs'] = [list(pair) for pair in intervals]
    return result
