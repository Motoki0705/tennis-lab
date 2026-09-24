"""COCO17/camera-local court encoding shared as code by independent PLCS models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.utils.models import TransformerBlockConfig
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES, FFNType


@dataclass(frozen=True)
class PersonModelConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 768
    num_stages: int = 4
    rope_dim: int = 32
    num_slots: int = 4
    dropout: float = 0.1
    ffn_type: FFNType = "swiglu"

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "num_heads", "ffn_dim", "num_stages", "rope_dim", "num_slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.hidden_dim % self.num_heads or self.rope_dim % 2 or self.rope_dim > self.hidden_dim // self.num_heads:
            raise ValueError("Invalid temporal attention dimensions")
        if not 0 <= self.dropout < 1 or self.ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError("Invalid dropout or FFN type")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> PersonModelConfig:
        if set(values) != set(asdict(cls())):
            raise ValueError("PLCS person model fields must exactly match its contract")
        return cls(**values)

    def attention(self) -> TransformerBlockConfig:
        return TransformerBlockConfig(dim=self.hidden_dim, n_heads=self.num_heads,
            ffn_dim=self.ffn_dim, head_dim=self.hidden_dim // self.num_heads,
            rope_dim=self.rope_dim, attn_dropout=self.dropout, attention_type="mha",
            n_kv_heads=None, rope_base=10000., ffn_type=self.ffn_type)


class PersonObservationEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.pose = nn.Sequential(nn.Linear(17 * 3, width), nn.GELU(), nn.Linear(width, width))
        self.court = nn.Linear(14 * 3, width)

    def forward(self, human_kp: Tensor, human_vis: Tensor, court_kp: Tensor, court_vis: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        visible = human_vis & ~padding_mask[..., None, None]
        court_visible = court_vis & ~padding_mask[..., None]
        pose = torch.cat((human_kp.masked_fill(~visible[..., None], 0).flatten(-2), visible.to(human_kp.dtype)), dim=-1)
        court = torch.cat((court_kp.masked_fill(~court_visible[..., None], 0).flatten(-2), court_visible.to(court_kp.dtype)), dim=-1)
        context = self.court(court).masked_fill(padding_mask[..., None], 0)
        tokens = self.pose(pose) + context[..., None, :]
        return tokens, visible.any(-1), context
