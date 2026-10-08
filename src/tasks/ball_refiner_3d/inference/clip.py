"""Whole-clip inference: every sequence is predicted in a single forward."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_refiner_3d.inference.flow_sampler import predict_sequences
from src.tasks.ball_refiner_3d.inference.segmentation import (
    DEFAULT_PICKING,
    EventPicking,
    predicted_segments,
)
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tasks.ball_refiner_3d.model_io.contracts import (
    PhysicsPrediction,
    RefinerPrediction,
    validate_input,
)
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel
from src.tasks.ball_refiner_3d.physics.reconstruction import integrate_segments
from src.tasks.ball_refiner_3d.physics.targets import FlightClock


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_normalized(
    model: RefinerModel,
    coordinates: Tensor,
    missing: Tensor,
    *,
    batch_size: int,
    seed: int,
    clock: FlightClock | None,
    segment: Tensor | None = None,
    picking: EventPicking = DEFAULT_PICKING,
) -> RefinerPrediction:
    """V,T,D -> V,T,D in ``batch_size`` sequences per forward, no windows.

    Physics heads need the ``clock`` of the training data.  Without a given
    ``segment (V,T)`` the flights are segmented at the predicted events.
    """
    validate_input(coordinates, missing, model.config.dimensions)
    if batch_size < 1 or seed < 0:
        raise ValueError("Require positive batch size and nonnegative seed")
    physics = model.config.physics_heads
    if physics != (clock is not None):
        raise ValueError("A flight clock is required exactly for physics heads")
    if segment is not None and (not physics or segment.shape != missing.shape):
        raise ValueError("A (V,T) segmentation requires physics heads")
    generator = torch.Generator(device=coordinates.device).manual_seed(seed)
    scale = coordinates.new_tensor(normalization(3)[0])
    parts: list[RefinerPrediction] = []
    was_training = model.training
    model.eval()
    try:
        for first in range(0, len(coordinates), batch_size):
            rows = slice(first, first + batch_size)
            output = predict_sequences(
                model, coordinates[rows], missing[rows], generator=generator
            )
            if not physics or clock is None:
                parts.append(
                    RefinerPrediction(output.coordinates, output.event_probability)
                )
                continue
            labels = (
                segment[rows]
                if segment is not None
                else torch.from_numpy(
                    np.stack(
                        [
                            predicted_segments(p, picking)
                            for p in output.event_probability.cpu().numpy()
                        ]
                    )
                ).to(coordinates.device)
            )
            output = predict_sequences(
                model,
                coordinates[rows],
                missing[rows],
                generator=generator,
                segment=labels,
            )
            if output.physics is None or output.physics.segment_states is None:
                raise RuntimeError("Physics heads returned no segment states")
            states = output.physics.segment_states
            integrated = integrate_segments(states, output.physics.field, labels, clock)
            parts.append(
                RefinerPrediction(
                    output.coordinates,
                    output.event_probability,
                    PhysicsPrediction(
                        output.physics.field,
                        output.physics.surface_logits.softmax(dim=-1),
                        labels,
                        states,
                        integrated / scale,
                    ),
                )
            )
    finally:
        model.train(was_training)
    result = _concatenate(parts)
    for value in (result.coordinates, result.event_probability):
        if not torch.isfinite(value).all():
            raise ValueError("Refiner produced nonfinite outputs")
    if result.physics is not None and not all(
        torch.isfinite(value).all()
        for value in (result.physics.integrated, result.physics.field)
    ):
        raise ValueError("Refiner produced nonfinite physics")
    return result


def _concatenate(parts: list[RefinerPrediction]) -> RefinerPrediction:
    coordinates = torch.cat([p.coordinates for p in parts])
    events = torch.cat([p.event_probability for p in parts])
    physics = [p.physics for p in parts]
    if all(value is None for value in physics):
        return RefinerPrediction(coordinates, events)
    present = [value for value in physics if value is not None]
    if len(present) != len(physics):
        raise RuntimeError("Physics outputs must be present for every batch")
    count = max(value.segment_states.shape[1] for value in present)
    return RefinerPrediction(
        coordinates,
        events,
        PhysicsPrediction(
            torch.cat([value.field for value in present]),
            torch.cat([value.surface_probability for value in present]),
            torch.cat([value.segment for value in present]),
            torch.cat(
                [
                    F.pad(
                        value.segment_states,
                        (0, 0, 0, count - value.segment_states.shape[1]),
                    )
                    for value in present
                ]
            ),
            torch.cat([value.integrated for value in present]),
        ),
    )
