"""Fuse four masked court-player slots into the pretrained frame-token stream."""

from __future__ import annotations

from typing import Literal, Self

import torch
from torch import Tensor, nn

from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.models.components.embeddings import coordinate_features
from src.tasks.ball_refiner_3d.models.generators.regression import RegressionRefiner
from src.tasks.ball_refiner_3d.post_training.motion import PLAYER_FEATURES, PLAYER_SLOTS

FusionMode = Literal["shared", "per-player", "set-attention"]
FUSION_MODES: tuple[FusionMode, ...] = ("shared", "per-player", "set-attention")


def _zero(linear: nn.Linear) -> nn.Linear:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


class PlayerEncoder(nn.Sequential):
    def __init__(self, width: int, *, zero_output: bool) -> None:
        output = nn.Linear(128, width)
        super().__init__(
            nn.Linear(PLAYER_FEATURES, 128),
            nn.RMSNorm(128),
            nn.SiLU(),
            _zero(output) if zero_output else output,
        )


class SetAttention(nn.Module):
    """One SAB followed by pooling with the original ball embedding as query."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.token_types = nn.Parameter(torch.zeros(2, width))
        nn.init.normal_(self.token_types, std=0.02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = _zero(nn.Linear(width, width))

    def forward(self, ball: Tensor, players: Tensor, valid: Tensor) -> Tensor:
        query = ball.flatten(0, 1).unsqueeze(1)
        tokens = torch.cat(
            (
                query + self.token_types[0],
                players.flatten(0, 1) + self.token_types[1],
            ),
            dim=1,
        )
        valid_players = valid.flatten(0, 1)
        ignore = torch.cat(
            (torch.zeros_like(valid_players[:, :1]), ~valid_players), dim=1
        )
        attended, _ = self.attention(
            tokens, tokens, tokens, key_padding_mask=ignore, need_weights=False
        )
        encoded = self.attention_norm(tokens + attended)
        encoded = self.ffn_norm(encoded + self.ffn(encoded))
        pooled, _ = self.pool(
            query, encoded, encoded, key_padding_mask=ignore, need_weights=False
        )
        residual = self.output(pooled[:, 0]).reshape_as(ball)
        return torch.where(valid.any(dim=-1, keepdim=True), residual, 0)


class PlayerFusion(nn.Module):
    def __init__(self, mode: FusionMode, width: int) -> None:
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"Unknown player fusion: {mode}")
        self.mode = mode
        self.encoders = nn.ModuleList(
            PlayerEncoder(width, zero_output=mode != "set-attention")
            for _ in range(PLAYER_SLOTS if mode == "per-player" else 1)
        )
        self.interaction = SetAttention(width) if mode == "set-attention" else None

    def forward(self, ball: Tensor, players: Tensor, valid: Tensor) -> Tensor:
        clean = torch.where(valid[..., None], players, 0)
        if self.mode == "per-player":
            encoded = torch.stack(
                [encoder(clean[:, :, i]) for i, encoder in enumerate(self.encoders)],
                dim=2,
            )
        else:
            encoded = self.encoders[0](clean)
        if self.interaction is not None:
            return ball + self.interaction(ball, encoded, valid)
        return ball + torch.where(valid[..., None], encoded, 0).sum(dim=2)


class MotionRefiner(nn.Module):
    """A frozen regression refiner with a trainable input-fusion residual."""

    def __init__(self, base: RegressionRefiner, mode: FusionMode) -> None:
        super().__init__()
        if base.config.physics_heads:
            raise ValueError(
                "Post-training requires the regression event checkpoint without physics heads"
            )
        self.base = base.requires_grad_(False).eval()
        self.fusion = PlayerFusion(mode, base.config.width)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        coordinates: Tensor,
        missing: Tensor,
        padding: Tensor,
        players: Tensor,
        player_valid: Tensor,
    ) -> RefinerOutput:
        ball = self.base.embed(coordinate_features(coordinates, missing))
        tokens = self.fusion(ball, players, player_valid & ~padding[..., None])
        refined = self.base.trunk_from_tokens(tokens, padding)
        return self.base.heads(refined, padding, segment=None)
