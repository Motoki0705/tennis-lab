"""Bidirectional coordinate regression and x0-parameterized conditional flow."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_refiner.coordinates.config import ModelConfig


def sinusoidal(values: Tensor, width: int) -> Tensor:
    frequency = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=values.dtype) * (-math.log(10000) / width))
    phase = values[..., None] * frequency
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def validate_input(coordinates: Tensor, missing: Tensor, dimensions: int) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != dimensions or coordinates.shape[1] < 1:
        raise ValueError(f"Coordinates must be B,T,{dimensions} with T>0")
    if missing.shape != coordinates.shape[:-1] or missing.dtype != torch.bool:
        raise ValueError("missing must be a B,T boolean tensor (true=missing)")
    if missing.device != coordinates.device or not coordinates.is_floating_point():
        raise ValueError("Coordinates and mask must share a device; coordinates must be floating point")
    if not torch.isfinite(coordinates[~missing]).all():
        raise ValueError("Observed coordinates must be finite")


class CoordinateRefiner(nn.Module):
    """No confidence, court, camera, or event inputs. Predict every frame."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.dimensions + 1 + (config.dimensions if config.architecture == "flow" else 0)
        self.input = nn.Linear(channels, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, dim_feedforward=4 * config.width,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, config.dimensions))
        self.flow_time = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width)) if config.architecture == "flow" else None

    def forward(self, coordinates: Tensor, missing: Tensor, state: Tensor | None = None, time: Tensor | None = None) -> Tensor:
        clean_input = torch.where(missing[..., None], 0, coordinates)
        features = [clean_input, missing[..., None].to(coordinates.dtype)]
        if self.config.architecture == "flow":
            if state is None or time is None or self.flow_time is None:
                raise ValueError("Flow forward requires its state and time")
            features.append(state)
        elif state is not None or time is not None:
            raise ValueError("Regression does not accept flow state/time")
        token = self.input(torch.cat(features, dim=-1))
        token = token + sinusoidal(torch.arange(coordinates.shape[1], device=coordinates.device, dtype=coordinates.dtype), self.config.width)[None]
        if self.flow_time is not None and time is not None:
            token = token + self.flow_time(sinusoidal(time * 1000, self.config.width))[:, None]
        # Missing frames remain queries and attend to both temporal directions.
        # Masked coordinates were removed above, so no ground truth can leak.
        return self.output(self.temporal(token))

    def flow_loss(self, coordinates: Tensor, missing: Tensor, target: Tensor, generator: torch.Generator) -> Tensor:
        if self.config.architecture != "flow":
            raise ValueError("flow_loss requires a flow model")
        time = torch.rand((len(target),), device=target.device, generator=generator) * 0.95
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        t = time[:, None, None]
        state = (1 - t) * noise + t * target
        x0 = self(coordinates, missing, state, time)
        # x0 parameterization of v=(x0-x_t)/(1-t); all target frames supervise it.
        return F.mse_loss((x0 - state) / (1 - t), target - noise)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def predict(self, coordinates: Tensor, missing: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        validate_input(coordinates, missing, self.config.dimensions)
        if self.config.architecture == "regression":
            return self(coordinates, missing)
        if generator is None:
            raise ValueError("Flow inference requires an explicit generator for reproducibility")
        state = torch.randn(coordinates.shape, dtype=coordinates.dtype, device=coordinates.device, generator=generator)
        for step in range(self.config.flow_steps):
            t = step / self.config.flow_steps
            time = torch.full((len(state),), t, dtype=coordinates.dtype, device=coordinates.device)
            x0 = self(coordinates, missing, state, time)
            state = state + (x0 - state) / (self.config.flow_steps - step)
        return state


class TrajectoryDiscriminator(nn.Module):
    """Conditional local/multiscale trajectory realism, including event motion."""

    def __init__(self, dimensions: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4 * dimensions + 1, width, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(width, 1, 3, padding=1),
        )

    def forward(self, trajectory: Tensor, coordinates: Tensor, missing: Tensor) -> Tensor:
        velocity = F.pad(torch.diff(trajectory, dim=1), (0, 0, 1, 0))
        acceleration = F.pad(torch.diff(velocity, dim=1), (0, 0, 1, 0))
        condition = torch.where(missing[..., None], 0, coordinates)
        features = torch.cat((trajectory, velocity, acceleration, condition, missing[..., None].to(trajectory.dtype)), -1)
        return self.net(features.transpose(1, 2))
