"""Untrained point estimators of the full saved 3D condition, validation only."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_rts import BallSmoothingConfig, smooth_mixture_mean
from .diffusion.dev_evaluation import RallyInput
from .diffusion.metrics import Array, TrajectoryMetrics


def condition_points(arrays: dict[str, Array]) -> dict[str, Array]:
    """Point summaries for baselines only; never change the model condition."""
    means = arrays['gmm3d_means_m'].astype(np.float64)
    weights = arrays['gmm3d_weights'].astype(np.float64)
    if (means.ndim != 3 or means.shape[-1] != 3 or weights.shape != means.shape[:2]
            or not np.isfinite(means).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.allclose(weights.sum(-1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Require the complete finite normalized mixture')
    # All components, including prior-only and float32 zero weights, remain.
    # No renormalization or threshold; ties in argmax use saved component order.
    return {'mixture_mean': np.einsum('tm,tmj->tj', weights, means),
            'top_component': means[np.arange(len(means)), weights.argmax(-1)]}


def evaluate_baselines(rallies: list[RallyInput], *, predictions: Path) -> dict[str, Any]:
    if not rallies or any(r.record['split'] != 'val' for r in rallies):
        raise ValueError('Baselines are restricted to validation rallies')
    predictions.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    collected = {name: TrajectoryMetrics() for name in ('mixture_mean', 'top_component', 'mixture_mean_rts', 'truth')}
    diagnostics = {}
    for rally in rallies:
        arrays = rally.arrays
        points = condition_points(arrays)
        dt = np.diff(arrays['timestamps_seconds'])
        if len(dt) < 3 or not (dt > 0).all() or not np.allclose(dt, dt[0], rtol=1e-8, atol=1e-12):
            raise ValueError('Baselines require uniform positive timestamps')
        points['mixture_mean_rts'], diagnostics[rally.record['rally_id']] = smooth_mixture_mean(points['mixture_mean'], fps=1 / float(dt[0]))
        points['truth'] = arrays['positions_3d_m']
        for name, value in points.items():
            collected[name].add(value[None], arrays)
        np.savez_compressed(predictions / (rally.record['rally_id'] + '.npz'), **points)
    return {'methods': {name: values.summarize() for name, values in collected.items()},
            'rallies': [r.record['rally_id'] for r in rallies], 'frames': sum(r.record['frames'] for r in rallies),
            'seconds': time.perf_counter() - started, 'rts_config': asdict(BallSmoothingConfig()),
            'rts_source_commit': '0f124818ea97dbec716e20a7fa6da315edb8ee82',
            'rts_diagnostics': diagnostics, 'tuning': 'none',
            'visibility': 'count of cameras with neither occlusion nor out_of_frame; not presence probability',
            'derivative_strata': 'original timeline; acceleration center, jerk left central frame; free-flight requires full stencil'}
