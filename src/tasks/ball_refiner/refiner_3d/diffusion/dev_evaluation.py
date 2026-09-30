"""Full-rally synthetic validation; no test records or checkpoint selection."""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from src.utils.schema.court_normalization import denormalize_court_position

from .data import rally_window
from .dev_config import DevConfig
from .flow import sample_trajectories, training_objective
from .metrics import Array, TrajectoryMetrics
from .model import TrajectoryDenoiser


@dataclass(frozen=True)
class RallyInput:
    record: dict[str, Any]
    arrays: dict[str, Array]


def evaluate_dev(
    model: TrajectoryDenoiser, rallies: list[RallyInput], config: DevConfig, *,
    device: str, objective: Literal['flow', 'regression'], check_budget: Callable[[], None],
    predictions: Path | None,
) -> dict[str, Any]:
    if not rallies or any(r.record['split'] != 'val' for r in rallies):
        raise ValueError('Evaluation is restricted to the complete validation split')
    started = time.perf_counter()
    collected = {name: TrajectoryMetrics() for name in ('mean', 'samples', 'truth')}
    loss_sums: dict[str, float] = defaultdict(float)
    total_frames = 0
    uncertainty = []
    was_training = model.training
    model.eval()
    if predictions is not None:
        predictions.mkdir(parents=True, exist_ok=False)
    try:
        with torch.no_grad():
            for rally in rallies:
                check_budget()
                record, arrays = rally.record, rally.arrays
                batch = rally_window(arrays, record, start=0, frames=record['frames'], allow_nonconverged=True).to(device)
                loss, terms = training_objective(model, batch, config.loss, torch.Generator(device=device).manual_seed(record['seed'] + 1), objective=objective)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                for key, value in {'loss': loss, **terms}.items():
                    loss_sums[key] += value.item() * record['frames']
                if objective == 'flow':
                    drawn = sample_trajectories(model, batch.condition, samples=config.samples, steps=config.steps,
                                                generator=torch.Generator(device=device).manual_seed(record['seed'] + 2))
                    trajectories = drawn.positions_m[:, 0].cpu().numpy()
                    covariance = drawn.covariance_m2[0].cpu().numpy()
                else:
                    state = torch.zeros_like(batch.target_positions_m)
                    predicted = model(state, torch.zeros(1, device=device), batch.condition)
                    trajectories = denormalize_court_position(predicted.positions_norm).cpu().numpy()
                    covariance = np.zeros((record['frames'], 3, 3), dtype=np.float32)
                mean = trajectories.mean(0)
                for kind, value in (('mean', mean[None]), ('samples', trajectories), ('truth', arrays['positions_3d_m'][None])):
                    collected[kind].add(value, arrays)
                uncertainty.append(np.sqrt(np.trace(covariance, axis1=-2, axis2=-1)))
                if predictions is not None:
                    fields = ('positions_3d_m', 'timestamps_seconds', 'occlusion_mask', 'out_of_frame_mask',
                              'event_region_mask', 'free_flight_mask', 'camera_true_K', 'camera_true_R', 'camera_true_t')
                    np.savez_compressed(predictions / (record['rally_id'] + '.npz'),
                                        samples_m=trajectories, mean_m=mean, covariance_m2=covariance,
                                        **{key: arrays[key] for key in fields})
                total_frames += record['frames']
                check_budget()
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started
    summaries = {kind: values.summarize() for kind, values in collected.items()}
    return {'rallies': len(rallies), 'frames': total_frames, 'seconds': elapsed,
            'frames_per_second': total_frames / elapsed,
            'loss': {key: value / total_frames for key, value in loss_sums.items()},
            'metrics': {kind: values['metrics'] for kind, values in summaries.items()},
            'by_visible_cameras': {kind: values['by_visible_cameras'] for kind, values in summaries.items()},
            'uncertainty_rms_radius_m': float(np.concatenate(uncertainty).mean()),
            'inference': 'whole rally, no stitching; fixed noise per rally/update; mean and all samples reported',
            'reprojection': 'source pixels vs synthetic truth on in-image GT; front-only errors plus explicit invalid-depth counts',
            'derivatives': 'm/s^2 and m/s^3; original timeline, full free-flight stencils; strata use acceleration center / jerk left center'}
