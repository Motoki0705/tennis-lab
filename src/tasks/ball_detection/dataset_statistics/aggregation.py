"""Pool original samples and expose independent distributions between clips."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contracts import Measurements, Samples
from .summaries import serialize, summarize

SUM_VIEWS = ('/occupancy', '/movement', '/speed_sum', '/edge_count',
             '/direction_x_sum', '/direction_y_sum', '/position_supervision', '/position_missing', '/seconds_total')


def aggregate(items: list[Measurements]) -> dict[str, Any]:
    pooled = Measurements()
    clip_distributions: dict[str, Any] = {}
    for key in sorted({k for item in items for k in item.counts}):
        values = np.asarray([item.counts[key] for item in items if key in item.counts], np.float64)
        pooled.counts[key] = int(values.sum())
        clip_distributions['counts/' + key] = dict(summarize(Samples(values, 'count', len(values))), sample_unit='clip')
    for key in sorted({k for item in items for k in item.rates}):
        pairs = [item.rates[key] for item in items if key in item.rates]
        pooled.rates[key] = (sum(n for n, _ in pairs), sum(d for _, d in pairs))
        values = np.asarray([n / d if d else np.nan for n, d in pairs])
        clip_distributions['rates/' + key] = dict(summarize(Samples(values, 'fraction', len(values))), sample_unit='clip')
    for key in sorted({k for item in items for k in item.samples}):
        samples = [item.samples[key] for item in items if key in item.samples]
        units = {sample.unit for sample in samples}
        if len(units) != 1:
            raise ValueError(f'Mixed units for {key}')
        pooled.samples[key] = Samples(np.concatenate([s.values for s in samples]), samples[0].unit, sum(s.eligible for s in samples))
        summaries = [summarize(s) for s in samples]
        for statistic in ('mean', 'median', 'p5', 'p95', 'min', 'max'):
            values = np.asarray([np.nan if summary[statistic] is None else summary[statistic] for summary in summaries])
            clip_distributions[f'{statistic}/{key}'] = dict(summarize(Samples(values, samples[0].unit, len(samples))), sample_unit='clip')
    for key in sorted({k for item in items for k in item.views if k.endswith(SUM_VIEWS)}):
        matrices = [np.asarray(item.views[key]) for item in items if key in item.views]
        pooled.views[key] = np.sum(matrices, axis=0).tolist()
    for key in sorted({k for item in items for k in item.views if k.endswith('/transitions')}):
        transitions: Counter[tuple[int, int]] = Counter()
        for item in items:
            for source, target, count in item.views[key]:
                transitions[source, target] += count
        pooled.views[key] = [[a, b, count] for (a, b), count in sorted(transitions.items())]
    return dict(pooled=serialize(pooled), between_clips=clip_distributions, clips=len(items))
