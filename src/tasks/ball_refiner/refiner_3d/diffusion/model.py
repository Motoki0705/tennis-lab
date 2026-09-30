"""Full-mixture conditioning and a temporal absolute-position denoiser."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.court_normalization import normalize_court_position


@dataclass(frozen=True)
class ModelConfig:
    width: int
    layers: int
    heads: int
    feedforward_multiplier: int
    time_frequencies: int
    dropout: float

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.width, self.layers, self.heads, self.feedforward_multiplier, self.time_frequencies)):
            raise ValueError("Model dimensions must be positive integers")
        if self.width % self.heads or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention heads/dropout")


@dataclass(frozen=True)
class MixtureCondition:
    means_m: Tensor  # B,T,M,3
    covariance_m2: Tensor  # B,T,M,3,3; includes all alternatives
    weights: Tensor  # B,T,M
    camera_subsets: Tensor  # B,T,M,3 bool; fixed three-view scaffold
    prior_only_probability: Tensor  # B,T
    timestamps_seconds: Tensor  # B,T
    padding_mask: Tensor  # B,T bool; actual missing observations are NOT padding

    def __post_init__(self) -> None:
        if self.means_m.ndim != 4 or self.means_m.shape[-1] != 3:
            raise ValueError("Mixture means must be B,T,M,3")
        b, t, m, _ = self.means_m.shape
        if min(b, m) < 1 or t < 3:
            raise ValueError("Need nonempty batch/mixture and at least three frames")
        specs = (
            ("means_m", (b, t, m, 3), torch.float32),
            ("covariance_m2", (b, t, m, 3, 3), torch.float32),
            ("weights", (b, t, m), torch.float32),
            ("camera_subsets", (b, t, m, 3), torch.bool),
            ("prior_only_probability", (b, t), torch.float32),
            ("timestamps_seconds", (b, t), torch.float32),
            ("padding_mask", (b, t), torch.bool),
        )
        for name, shape, dtype in specs:
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype or value.device != self.means_m.device:
                raise ValueError(f"Invalid condition tensor {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite condition tensor {name}")
        valid = ~self.padding_mask
        if bool((valid.sum(-1) < 3).any()) or bool((self.padding_mask[:, :-1] & valid[:, 1:]).any()):
            raise ValueError("Need >=3 real frames with right padding only")
        delta = self.timestamps_seconds[:, 1:] - self.timestamps_seconds[:, :-1]
        if bool((delta[valid[:, 1:]] <= 0).any()):
            raise ValueError("Real timestamps must increase")
        if bool((self.weights < 0).any()) or not torch.allclose(self.weights.sum(-1)[valid], torch.ones_like(self.weights.sum(-1)[valid]), atol=1e-5):
            raise ValueError("Real-frame mixture mass must sum to one")
        if bool(((self.prior_only_probability < 0) | (self.prior_only_probability > 1)).any()):
            raise ValueError("Invalid prior-only probability")
        cov = self.covariance_m2[valid]
        if not torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6):
            raise ValueError("Covariance must be symmetric")
        torch.linalg.cholesky(cov)

    def to(self, device: str | torch.device) -> MixtureCondition:
        return MixtureCondition(**{field.name: getattr(self, field.name).to(device) for field in fields(self)})


@dataclass(frozen=True)
class DenoiserOutput:
    positions_norm: Tensor
    event_logits: Tensor  # hit/bounce are two independent classes


def condition_features(condition: MixtureCondition) -> Tensor:
    """The exact normalized per-component features used by the denoiser."""
    scale = condition.means_m.new_tensor(COURT_COORD_SCALE_XYZ)
    covariance_norm = condition.covariance_m2 / (scale[:, None] * scale[None, :])
    return torch.cat((
        normalize_court_position(condition.means_m), covariance_norm.flatten(-2),
        condition.camera_subsets.to(condition.means_m.dtype),
    ), dim=-1)


def validate_flow_state(noisy_positions_norm: Tensor, flow_time: Tensor, condition: MixtureCondition) -> None:
    """Validate at the train/sample boundary, outside computation-only forward."""
    b, t = condition.padding_mask.shape
    if noisy_positions_norm.shape != (b, t, 3) or flow_time.shape != (b,):
        raise ValueError("Invalid noisy trajectory or flow time shape")
    if not bool(torch.isfinite(noisy_positions_norm).all()) or not bool(torch.isfinite(flow_time).all()) or bool(((flow_time < 0) | (flow_time > 1)).any()):
        raise ValueError("Flow state must be finite with time in [0,1]")


class TrajectoryDenoiser(nn.Module):
    """Predict x0 itself. There is no offset addition to a triangulated path."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), config.time_frequencies)))
        self.component_encoder = nn.Sequential(nn.Linear(15, config.width), nn.SiLU(), nn.Linear(config.width, config.width), nn.SiLU())
        self.state_encoder = nn.Linear(4 * config.time_frequencies + 4, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * config.feedforward_multiplier,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.position_head = nn.Linear(config.width, 3)
        self.event_head = nn.Linear(config.width, 2)

    def time_features(self, value: Tensor) -> Tensor:
        angles = value[..., None] * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def encode_condition(self, condition: MixtureCondition) -> Tensor:
        """Expose the existing pooled tokens for frozen read-out diagnostics."""
        components = self.component_encoder(condition_features(condition))
        # Nonlinear component encoding precedes pooling: not moment matching.
        return (components * condition.weights[..., None]).sum(dim=-2)

    def forward(self, noisy_positions_norm: Tensor, flow_time: Tensor, condition: MixtureCondition) -> DenoiserOutput:
        b, t, _, _ = condition.means_m.shape
        context = self.encode_condition(condition)
        state = torch.cat((
            noisy_positions_norm,
            self.time_features(flow_time)[:, None].expand(b, t, -1),
            self.time_features(condition.timestamps_seconds),
            condition.prior_only_probability[..., None],
        ), dim=-1)
        hidden = context + self.state_encoder(state)
        hidden = hidden.masked_fill(condition.padding_mask[..., None], 0)
        hidden = self.norm(self.temporal(hidden, src_key_padding_mask=condition.padding_mask))
        positions = self.position_head(hidden).masked_fill(condition.padding_mask[..., None], 0)
        events = self.event_head(hidden).masked_fill(condition.padding_mask[..., None], 0)
        return DenoiserOutput(positions, events)
