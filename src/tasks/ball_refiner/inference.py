"""Annotation-free single-camera inference shared by evaluation and deployment."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.data.inputs import collate_inputs, input_to, slice_input
from src.tasks.ball_refiner.data.temporal import window_owners, window_starts
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.base.model_io import BoundModelIO

RefinerPair: TypeAlias = BoundModelIO[Refiner2DInput, torch.Tensor, BallGMM2D]


@dataclass(frozen=True)
class SequencePrediction:
    """Complete CPU GMM and the exact source window chosen for every real frame."""

    distribution: BallGMM2D
    window_start: NDArray[np.int64]
    time_index: NDArray[np.int64]
    window_length: int

    def __post_init__(self) -> None:
        batch, frames = self.distribution.means.shape[:2]
        if batch != 1 or self.distribution.means.device.type != "cpu":
            raise ValueError("Sequence prediction must be one complete camera on CPU")
        if type(self.window_length) is not int or not 1 <= self.window_length <= frames:
            raise ValueError("Invalid real prediction window length")
        for value in (self.window_start, self.time_index):
            if value.dtype != np.int64 or value.shape != (frames,):
                raise ValueError("Prediction provenance must address every source frame")
        if ((self.window_start < 0).any() or (self.window_start + self.window_length > frames).any()
                or (self.time_index < 0).any() or (self.time_index >= self.window_length).any()
                or not np.array_equal(self.window_start + self.time_index, np.arange(frames))):
            raise ValueError("Prediction provenance does not address real source frames")


def predict_sequence(
    pair: RefinerPair, inputs: Refiner2DInput, *, window_length: int, stride: int,
    batch_size: int, device: torch.device,
) -> SequencePrediction:
    """Copy all GMM fields atomically from the nearest-centre window, earlier on ties.

    Inputs contain one camera's entire CPU timeline, including explicit pose/court
    if configured. No labels, frame selection by detections, time padding, component
    averaging, presence threshold or fallback is applied. The caller places the model
    on the requested device; only each inference batch moves there.
    """
    if any(type(value) is not int or value < 1 for value in (window_length, stride, batch_size)):
        raise ValueError("Inference window length, stride and batch size must be positive integers")
    times = inputs.timestamps_seconds
    if times.ndim != 2 or times.shape[0] != 1 or times.device.type != "cpu":
        raise ValueError("Sequence inference requires one camera's complete CPU timeline")
    frames = times.shape[1]
    starts = window_starts(frames, window_length, stride)
    owners = window_owners(frames, starts, window_length)
    # Validate the global timeline too: a discontinuity could otherwise sit between
    # disjoint windows and escape each batch's strictly-increasing time check.
    with torch.no_grad():
        pair.build_call(inputs)
    collected: dict[str, torch.Tensor] = {}
    was_training = pair.model.training
    try:
        pair.model.eval()
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                selected = starts[offset:offset + batch_size]
                batch = collate_inputs([slice_input(inputs, start, start + window_length) for start in selected])
                prediction = pair.run(input_to(batch, device))
                for field in fields(prediction):
                    values = getattr(prediction, field.name).cpu()
                    if values.shape[:2] != (len(selected), window_length):
                        raise ValueError("Model output does not cover the requested real windows")
                    if field.name not in collected:
                        collected[field.name] = torch.empty((1, frames, *values.shape[2:]), dtype=values.dtype)
                    for row, start in enumerate(selected):
                        indices = np.flatnonzero(owners[start:start + window_length] == offset + row)
                        collected[field.name][0, start + indices] = values[row, indices]
    finally:
        pair.model.train(was_training)
    selected_starts = np.asarray(starts, dtype=np.int64)[owners]
    return SequencePrediction(
        BallGMM2D(**collected), selected_starts, np.arange(frames, dtype=np.int64) - selected_starts, window_length,
    )
