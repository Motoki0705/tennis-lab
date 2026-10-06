"""Bidirectional RoPE/SwiGLU generator shared by 2D, 3D and conditional flow."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.ball_refiner.coordinates.config import ModelConfig
from src.tasks.ball_refiner.coordinates.models.contracts import (
    sinusoidal,
    validate_input,
)
from src.utils.models.components import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)


class CoordinateRefiner(nn.Module):
    """No confidence, court, camera, or event inputs. Predict every frame."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.dimensions + 1 + (config.dimensions if config.architecture == "flow" else 0)
        self.input = nn.Linear(channels, config.width)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(TransformerBlockConfig(
                dim=config.width, n_heads=config.heads, ffn_dim=config.ffn_dim,
                head_dim=config.width // config.heads, rope_dim=config.rope_dim,
                attn_dropout=config.dropout, attention_type="mha", n_kv_heads=None,
                rope_base=config.rope_theta, ffn_type="swiglu",
            )) for _ in range(config.layers)
        ])
        self.output = nn.Sequential(RMSNorm(config.width), nn.Linear(config.width, config.dimensions))
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            dim=config.rope_dim, seqlen=config.window_length, base=config.rope_theta,
        ), persistent=False)
        self.flow_time = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width)) if config.architecture == "flow" else None
        self.register_forward_pre_hook(self._validate_flow_arguments, with_kwargs=True)

    def _validate_flow_arguments(self, _module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """Validate call arity before entering the computation-only forward."""
        coordinates = args[0] if args else kwargs.get("coordinates")
        if isinstance(coordinates, Tensor) and coordinates.shape[1] > self.config.window_length:
            raise ValueError("Sequence exceeds configured window_length; use windowed inference")
        state = args[2] if len(args) > 2 else kwargs.get("state")
        time = args[3] if len(args) > 3 else kwargs.get("time")
        if self.config.architecture == "flow":
            if state is None or time is None or self.flow_time is None:
                raise ValueError("Flow forward requires its state and time")
        elif state is not None or time is not None:
            raise ValueError("Regression does not accept flow state/time")

    def forward(self, coordinates: Tensor, missing: Tensor, state: Tensor | None = None, time: Tensor | None = None) -> Tensor:
        clean_input = torch.where(missing[..., None], 0, coordinates)
        features = [clean_input, missing[..., None].to(coordinates.dtype)]
        if self.config.architecture == "flow":
            features.append(cast(Tensor, state))
        token = self.input(torch.cat(features, dim=-1))
        token = self.input_dropout(token)
        if self.flow_time is not None and time is not None:
            token = token + self.flow_time(sinusoidal(time * 1000, self.config.width))[:, None]
        # Missing frames remain queries and attend to both temporal directions.
        # Masked coordinates were removed above, so no ground truth can leak.
        length = coordinates.shape[1]
        keep = torch.ones((len(coordinates), length, length), dtype=torch.bool, device=coordinates.device)
        frequencies = self.get_buffer("freqs_cis")[:length]
        for block in self.blocks:
            token = block(token, freqs_cis=frequencies, attn_mask=keep)
        return self.output(token)

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

