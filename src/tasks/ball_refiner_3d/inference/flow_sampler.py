"""Explicit-seed single-trajectory inference, independent of network forward."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput, validate_input
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel, bind_refiner
from src.tasks.ball_refiner_3d.models.generators.flow import FlowRefiner
from src.tasks.ball_refiner_3d.models.generators.regression import RegressionRefiner


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_sequences(
    model: RefinerModel,
    coordinates: Tensor,
    missing: Tensor,
    *,
    generator: torch.Generator | None,
    segment: Tensor | None = None,
) -> RefinerOutput:
    """One forward over whole (unpadded) sequences ``(B,T,3)``."""
    validate_input(coordinates, missing, 3)
    binding = bind_refiner(model)
    padding = torch.zeros_like(missing)
    if isinstance(model, RegressionRefiner):
        batch = {"coordinates": coordinates, "missing": missing, "padding": padding}
        if segment is not None:
            batch["segment"] = segment
        return cast(RefinerOutput, binding.run(batch))
    if not isinstance(model, FlowRefiner):
        raise TypeError("Unsupported refiner model")
    if segment is not None:
        raise ValueError("Flow refiners have no physics heads")
    if generator is None:
        raise ValueError(
            "Flow inference requires an explicit generator for reproducibility"
        )
    state = torch.randn(
        coordinates.shape,
        dtype=coordinates.dtype,
        device=coordinates.device,
        generator=generator,
    )
    for step in range(model.config.flow_steps):
        time = torch.full(
            (len(state),),
            step / model.config.flow_steps,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        result = binding.run(
            {
                "coordinates": coordinates,
                "missing": missing,
                "padding": padding,
                "state": state,
                "time": time,
            }
        )
        state = state + (result.coordinates - state) / (model.config.flow_steps - step)
    return RefinerOutput(state, result.event_logits)
