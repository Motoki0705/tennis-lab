"""Exact finite-sample summaries; empty distributions are unknown, not zero."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import Measurements, Samples


def summarize(sample: Samples) -> dict[str, Any]:
    values = sample.values[np.isfinite(sample.values)]
    if sample.eligible < len(values):
        raise ValueError('Valid sample count exceeds eligible count')
    result: dict[str, Any] = dict(n=len(values), eligible=sample.eligible, missing=sample.eligible - len(values), unit=sample.unit)
    result.update({k: None for k in ('mean', 'median', 'p5', 'p95', 'min', 'max')})
    if len(values):
        quantiles = np.asarray(np.quantile(values, [.05, .5, .95]))
        result.update(mean=float(values.mean()), median=float(quantiles[1]), p5=float(quantiles[0]),
                      p95=float(quantiles[2]), min=float(values.min()), max=float(values.max()))
    return result


def serialize(measurements: Measurements) -> dict[str, Any]:
    return dict(
        counts=measurements.counts,
        rates={key: dict(numerator=n, denominator=d, value=n / d if d else None)
               for key, (n, d) in measurements.rates.items()},
        distributions={key: summarize(sample) for key, sample in measurements.samples.items()},
        views=measurements.views, findings=measurements.findings,
    )
