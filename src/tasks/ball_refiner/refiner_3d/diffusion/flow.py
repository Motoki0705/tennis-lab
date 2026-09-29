"""Straight conditional flow paths with absolute-x0 prediction and Euler sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_3d.diffusion.losses import (
    LossConfig,
    TrainingBatch,
    trajectory_loss,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    MixtureCondition,
    TrajectoryDenoiser,
    validate_flow_state,
)
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
)


def training_objective(
    model: TrajectoryDenoiser, batch: TrainingBatch, config: LossConfig,
    generator: torch.Generator, *, objective: Literal["flow", "regression"],
) -> tuple[Tensor, dict[str, Tensor]]:
    clean = normalize_court_position(batch.target_positions_m)
    b = clean.shape[0]
    if objective == "flow":
        time = torch.rand(b, device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        state = (1 - time[:, None, None]) * noise + time[:, None, None] * clean
    elif objective == "regression":
        # Same architecture, one call, no target/noisy-path information as input.
        time = torch.zeros(b, device=clean.device)
        state = torch.zeros_like(clean)
    else:
        raise ValueError(f"Unknown training objective: {objective}")
    validate_flow_state(state, time, batch.condition)
    output = model(state, time, batch.condition)
    loss, terms = trajectory_loss(output, batch, config)
    return loss, terms


@dataclass(frozen=True)
class TrajectorySamples:
    positions_m: Tensor  # S,B,T,3
    mean_m: Tensor  # B,T,3
    covariance_m2: Tensor  # B,T,3,3; empirical variation across samples


def sample_trajectories(
    model: TrajectoryDenoiser, condition: MixtureCondition, *,
    samples: int, steps: int, generator: torch.Generator,
) -> TrajectorySamples:
    if samples < 2 or steps < 1:
        raise ValueError("Need >=2 samples for uncertainty and >=1 ODE steps")
    b, t = condition.padding_mask.shape
    trajectories = []
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(samples):
                state = torch.randn((b, t, 3), device=condition.means_m.device, generator=generator)
                for step in range(steps):
                    time = torch.full((b,), step / steps, device=state.device)
                    validate_flow_state(state, time, condition)
                    x0 = model(state, time, condition).positions_norm
                    # v_t = (x0 - x_t)/(1-t), with no evaluation at singular t=1.
                    state = state + (x0 - state) / (steps - step)
                trajectories.append(denormalize_court_position(state))
    finally:
        model.train(training)
    positions = torch.stack(trajectories)
    mean = positions.mean(0)
    centered = positions - mean
    covariance = torch.einsum("sbti,sbtj->btij", centered, centered) / (samples - 1)
    return TrajectorySamples(positions, mean, covariance)
