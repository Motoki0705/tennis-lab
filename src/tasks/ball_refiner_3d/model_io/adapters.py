"""Coordinate normalization and tensor boundary adapters."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput, validate_input
from src.tasks.base.model_io import ModelCall


def normalization(dimensions: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if dimensions == 3:
        return np.array([10, 20, 5], np.float32), np.zeros(3, np.float32)
    raise ValueError("Require 3D coordinates")


class RefinerAdapter:
    """Validate normalized tensors before compiled forward; GT stays outside calls."""

    def __init__(
        self, model_type: type[nn.Module], *, window_length: int, flow: bool
    ) -> None:
        self._model_type = model_type
        self.window_length = window_length
        self.flow = flow

    @property
    def model_type(self) -> type[nn.Module]:
        return self._model_type

    def build_call(self, batch: dict[str, Tensor]) -> ModelCall:
        coordinates, missing = batch["coordinates"], batch["missing"]
        validate_input(coordinates, missing, 3)
        if coordinates.shape[1] > self.window_length:
            raise ValueError(
                "Sequence exceeds configured window_length; use windowed inference"
            )
        if self.flow:
            if "state" not in batch or "time" not in batch:
                raise ValueError("Flow requires both state and time")
            state, time = batch["state"], batch["time"]
            if state.shape != coordinates.shape or time.shape != (len(coordinates),):
                raise ValueError("Flow requires B,T,3 state and B time")
            if (
                state.device != coordinates.device
                or time.device != coordinates.device
                or not torch.isfinite(state).all()
                or not torch.isfinite(time).all()
            ):
                raise ValueError(
                    "Flow state/time must be finite and share the input device"
                )
            return ModelCall(args=(coordinates, missing, state, time))
        if "state" in batch or "time" in batch:
            raise ValueError("Regression does not accept flow state/time")
        return ModelCall(args=(coordinates, missing))

    def decode_output(self, output: RefinerOutput) -> RefinerOutput:
        if (
            not isinstance(output, RefinerOutput)
            or output.coordinates.ndim != 3
            or output.coordinates.shape[-1] != 3
            or output.event_logits.shape != (*output.coordinates.shape[:-1], 2)
        ):
            raise ValueError("Require B,T,3 coordinates and B,T,2 event logits")
        return output
