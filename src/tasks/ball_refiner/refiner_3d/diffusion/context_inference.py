"""Full-frame context inference shared by fixed-weight probes and validation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Literal

import torch
from torch import Tensor

from src.utils.schema.court_normalization import denormalize_court_position

from .data import window_starts
from .flow import sample_trajectories
from .losses import TrainingBatch
from .model import DenoiserOutput, MixtureCondition, TrajectoryDenoiser


def _window(value: Tensor, start: int, frames: int, *, axis: int) -> Tensor:
    shape = list(value.shape)
    shape[axis] = frames
    result = value.new_zeros(shape)
    length = min(frames, value.shape[axis] - start)
    result.narrow(axis, 0, length).copy_(value.narrow(axis, start, length))
    return result


def predict_context(
    model: TrajectoryDenoiser, batch: TrainingBatch, *, objective: Literal['flow', 'regression'],
    initial_noise: Tensor, probe_state: Tensor, probe_time: Tensor, steps: int,
    frames: int | None, check_budget: Callable[[], None],
) -> tuple[Tensor, DenoiserOutput, list[dict[str, int]]]:
    """Keep earliest-window ownership, absolute timestamps, and every real frame.

    The unpadded input is one whole rally. Padding matches training windows.
    Metrics/loss are computed on the assembled original timeline, including seams.
    """
    condition = batch.condition
    if condition.padding_mask.shape[0] != 1 or bool(condition.padding_mask.any()):
        raise ValueError('Context probe requires one complete unpadded rally')
    length = condition.padding_mask.shape[1]
    width = length if frames is None else frames
    starts = [0] if frames is None else window_starts(length, width, width)
    samples = initial_noise.shape[0] if objective == 'flow' else 1
    prediction = condition.means_m.new_empty((samples, length, 3))
    probe_positions = torch.empty_like(batch.target_positions_m)
    probe_events = condition.means_m.new_empty((1, length, 2))
    covered = 0
    ownership = []
    with torch.no_grad():
        for start in starts:
            check_budget()
            stop = min(length, start + width)
            if start > covered:
                raise ValueError('Context windows left uncovered frames')
            values = {f.name: _window(getattr(condition, f.name), start, width, axis=1)
                      for f in fields(MixtureCondition)}
            values['padding_mask'][:, stop - start:] = True
            selected = MixtureCondition(**values)
            probe = model(_window(probe_state, start, width, axis=1), probe_time, selected)
            if objective == 'flow':
                trajectories = sample_trajectories(
                    model, selected, samples=samples, steps=steps,
                    generator=torch.Generator().manual_seed(0),
                    initial_noise=_window(initial_noise, start, width, axis=2),
                ).positions_m[:, 0]
            else:
                trajectories = denormalize_court_position(probe.positions_norm)
            offset = covered - start
            prediction[:, covered:stop] = trajectories[:, offset:stop - start]
            probe_positions[:, covered:stop] = probe.positions_norm[:, offset:stop - start]
            probe_events[:, covered:stop] = probe.event_logits[:, offset:stop - start]
            ownership.append({'window_start': start, 'real_stop': stop, 'owned_start': covered,
                              'owned_stop': stop, 'padded_frames': width - (stop - start)})
            covered = stop
    if covered != length or not bool(torch.isfinite(prediction).all()):
        raise ValueError('Incomplete/nonfinite context predictions')
    return prediction, DenoiserOutput(probe_positions, probe_events), ownership
