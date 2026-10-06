from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..contracts import ClipInput, Measurements
from ..scopes import runs


def compute(data: ClipInput, scope: NDArray[np.bool_]) -> Measurements:
    result = Measurements()
    masks = {'coordinate_gap': data.states.located == 0,
             'observed_gap': data.states.kinds[:, 0] == 0,
             'located_run': data.states.located > 0}
    for name, mask in masks.items():
        for boundary_name, boundaries in [('raw', np.zeros(len(scope), bool)), ('bounded', data.breaks)]:
            intervals = runs(mask & scope, boundaries)
            key = f'gaps/{name}/{boundary_name}'
            result.counts[key] = len(intervals)
            result.sample(key + '/frames', [b - a for a, b in intervals], 'frames')
            result.sample(key + '/seconds', [data.durations[a:b].sum() for a, b in intervals], 'seconds')
            result.views[key] = [list(pair) for pair in intervals]
    return result
