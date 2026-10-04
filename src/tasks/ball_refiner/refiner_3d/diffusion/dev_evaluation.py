"""Synthetic validation with an explicit inference context and complete frames."""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
)

from .context_inference import predict_context
from .data import rally_window
from .dev_config import DevConfig
from .flow import sample_trajectories, training_objective
from .losses import TrainingBatch, trajectory_loss
from .metrics import Array, TrajectoryMetrics
from .model import TrajectoryDenoiser


@dataclass(frozen=True)
class RallyInput:
    record: dict[str, Any]
    arrays: dict[str, Array]


def _evaluate_windowed(
    model: TrajectoryDenoiser, batch: TrainingBatch, config: DevConfig, *,
    seed: int, objective: Literal['flow', 'regression'], check_budget: Callable[[], None],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Array, Array, list[dict[str, int]]]:
    clean = normalize_court_position(batch.target_positions_m)
    device = clean.device
    sample_rng = torch.Generator(device=device).manual_seed(seed + 2)
    noise = torch.stack([torch.randn(clean.shape, device=device, generator=sample_rng)
                         for _ in range(config.samples)])
    if objective == 'flow':
        probe_rng = torch.Generator(device=device).manual_seed(seed + 1)
        time = torch.rand(1, device=device, generator=probe_rng)
        state = (1 - time[:, None, None]) * torch.randn(clean.shape, device=device, generator=probe_rng) + time[:, None, None] * clean
    else:
        time, state = torch.zeros(1, device=device), torch.zeros_like(clean)
    trajectories, probe, ownership = predict_context(
        model, batch, objective=objective, initial_noise=noise, probe_state=state,
        probe_time=time, steps=config.steps, frames=config.validation_frames, check_budget=check_budget,
    )
    # Derivatives and loss include window seams on the assembled original timeline.
    loss, terms = trajectory_loss(probe, batch, config.loss)
    centered = trajectories - trajectories.mean(0)
    covariance = (torch.einsum('sti,stj->tij', centered, centered) / (config.samples - 1)
                  if objective == 'flow' else clean.new_zeros((clean.shape[1], 3, 3)))
    return loss, terms, trajectories.cpu().numpy(), covariance.cpu().numpy(), ownership


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
    windows = {}
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
                if config.validation_frames is not None:
                    loss, terms, trajectories, covariance, ownership = _evaluate_windowed(
                        model, batch, config, seed=record['seed'], objective=objective, check_budget=check_budget,
                    )
                    windows[record['rally_id']] = ownership
                else:
                    loss, terms = training_objective(model, batch, config.loss, torch.Generator(device=device).manual_seed(record['seed'] + 1), objective=objective)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                for key, value in {'loss': loss, **terms}.items():
                    loss_sums[key] += value.item() * record['frames']
                if config.validation_frames is None and objective == 'flow':
                    drawn = sample_trajectories(model, batch.condition, samples=config.samples, steps=config.steps,
                                                generator=torch.Generator(device=device).manual_seed(record['seed'] + 2))
                    trajectories = drawn.positions_m[:, 0].cpu().numpy()
                    covariance = drawn.covariance_m2[0].cpu().numpy()
                elif config.validation_frames is None:
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
            'validation_frames': config.validation_frames, 'windows': windows,
            'inference': ('whole rally, no stitching' if config.validation_frames is None else
                          f'T{config.validation_frames}/stride{config.validation_frames}, absolute times, right padding, earlier window owns overlap')
                         + '; full-rally device noise seed+2 sliced by frame; probe seed+1; fixed across updates; mean and all samples reported',
            'reprojection': 'source pixels vs synthetic truth on in-image GT; front-only errors plus explicit invalid-depth counts',
            'derivatives': 'm/s^2 and m/s^3; original timeline, full free-flight stencils; strata use acceleration center / jerk left center'}
