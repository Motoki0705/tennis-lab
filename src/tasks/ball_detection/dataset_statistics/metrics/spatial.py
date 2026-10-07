from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..configuration import StatisticsConfig
from ..contracts import ClipInput, Measurements


def normalized_xy(data: ClipInput) -> NDArray[np.float64]:
    # This is the same source endpoint normalization as the model's targets.
    return data.states.xy / (data.clip.scale * np.array([data.clip.source_width - 1, data.clip.source_height - 1]))


def compute(data: ClipInput, scope: NDArray[np.bool_], config: StatisticsConfig) -> Measurements:
    result = Measurements()
    positions = normalized_xy(data)
    for name, eligible in [('observed', data.states.visible_observation),
                           ('located_single', data.states.reviewed & (data.states.count == 1) & (data.states.located == 1))]:
        values = positions[scope & eligible]
        prefix = f'spatial/{name}'
        result.sample(prefix + '/x', values[:, 0], 'image_width', int(scope.sum()))
        result.sample(prefix + '/y', values[:, 1], 'image_height', int(scope.sum()))
        result.sample(prefix + '/edge_distance', np.min(np.c_[values, 1 - values], axis=1), 'normalized_axis')
        for axis, label in enumerate(('x', 'y')):
            spread = np.diff(np.quantile(values[:, axis], [.05, .95])) if len(values) else []
            result.sample(prefix + f'/{label}_p95_p5', spread, 'normalized_axis', 1)
        histogram, _, _ = np.histogram2d(values[:, 1], values[:, 0], bins=(config.grid_y, config.grid_x), range=((0, 1), (0, 1)))
        result.views[prefix + '/occupancy'] = histogram.astype(int).tolist()
        result.counts[prefix + '/outside_endpoint_grid'] = int(((values < 0) | (values > 1)).any(axis=1).sum())
    return result
