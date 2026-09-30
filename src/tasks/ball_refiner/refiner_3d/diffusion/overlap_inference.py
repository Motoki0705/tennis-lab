"""Explicit triangular overlap inference; no change to training/default stitching."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Literal

import torch
from torch import Tensor

from src.utils.schema.court_normalization import denormalize_court_position

from .context_inference import _window
from .data import window_starts
from .flow import sample_trajectories
from .losses import TrainingBatch
from .model import MixtureCondition, TrajectoryDenoiser


def predict_overlap(
    model: TrajectoryDenoiser, batch: TrainingBatch, *, objective: Literal['flow', 'regression'],
    initial_noise: Tensor, steps: int, frames: int, stride: int,
    check_budget: Callable[[], None],
) -> tuple[Tensor, list[dict[str, int]]]:
    """Blend every real-frame prediction with min(i+1, frames-i)/(frames/2).

    Absolute timestamps and sample/frame noise are sliced before each prediction.
    Positive endpoint weights cover short rallies/tails; padding never gets weight.
    There is no GT/event-dependent weighting or sample/mixture selection.
    """
    condition = batch.condition
    if (condition.padding_mask.shape[0] != 1 or bool(condition.padding_mask.any())
            or objective not in ('flow', 'regression')):
        raise ValueError('Overlap requires one unpadded rally and an explicit objective')
    length = condition.padding_mask.shape[1]
    starts = window_starts(length, frames, stride)
    if stride >= frames:
        raise ValueError('Overlap requires stride smaller than frames')
    if (initial_noise.ndim != 4 or initial_noise.shape[1:] != (1, length, 3)
            or initial_noise.shape[0] < 2 or not bool(torch.isfinite(initial_noise).all())):
        raise ValueError('Need finite full-rally initial noise for every sample/frame')
    samples = initial_noise.shape[0] if objective == 'flow' else 1
    prediction = condition.means_m.new_zeros((samples, length, 3))
    total_weight = condition.means_m.new_zeros(length)
    index = torch.arange(frames, device=prediction.device, dtype=prediction.dtype)
    weights = torch.minimum(index + 1, frames - index) / (frames / 2)
    windows = []
    with torch.no_grad():
        for start in starts:
            check_budget()
            stop = min(length, start + frames)
            real = stop - start
            values = {f.name: _window(getattr(condition, f.name), start, frames, axis=1)
                      for f in fields(MixtureCondition)}
            values['padding_mask'][:, real:] = True
            selected = MixtureCondition(**values)
            if objective == 'flow':
                trajectories = sample_trajectories(
                    model, selected, samples=samples, steps=steps,
                    generator=torch.Generator().manual_seed(0),
                    initial_noise=_window(initial_noise, start, frames, axis=2),
                ).positions_m[:, 0]
            else:
                state = prediction.new_zeros((1, frames, 3))
                output = model(state, prediction.new_zeros(1), selected)
                trajectories = denormalize_court_position(output.positions_norm)
            if not bool(torch.isfinite(trajectories[:, :real]).all()):
                raise FloatingPointError('Nonfinite real-frame overlap prediction')
            prediction[:, start:stop] += trajectories[:, :real] * weights[:real, None]
            total_weight[start:stop] += weights[:real]
            windows.append({'window_start': start, 'real_stop': stop, 'padded_frames': frames - real})
    if not bool((total_weight > 0).all()):
        raise ValueError('Overlap left uncovered real frames')
    prediction /= total_weight[:, None]
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError('Nonfinite normalized overlap prediction')
    return prediction, windows
