from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.models.components.ffn_layers import default_ffn_dim


@dataclass(frozen=True)
class MDDPretrainConfig:
    stem_channels: tuple[int, int, int, int]
    mixed_channels: tuple[int, int]
    residual_blocks: tuple[int, int, int, int, int, int]
    decoder_channels: int
    dim: int
    heads: int
    layers: int
    ffn_dim: int
    dropout: float
    rope_base: float
    activation_checkpointing: bool
    encoder_variant: str
    temporal_mixing: str

    def __post_init__(self) -> None:
        if self.encoder_variant not in {"residual", "convnext_v2", "fasternet"}:
            raise ValueError("Unknown CNN encoder variant")
        if self.temporal_mixing not in {"dense3d", "factorized"}:
            raise ValueError("Unknown CNN temporal mixing")
        if len(self.stem_channels) != 4 or len(self.mixed_channels) != 2 or len(self.residual_blocks) != 6:
            raise ValueError("Require four stem, two mixed stages and six residual depths")
        if any(type(v) is not int or v < 1 for v in (*self.stem_channels, *self.mixed_channels, self.decoder_channels)):
            raise ValueError("CNN channels must be positive integers")
        if any(type(v) is not int or v < 0 for v in self.residual_blocks):
            raise ValueError("Residual depths must be nonnegative integers")
        if self.heads < 1 or self.dim < 4 or self.dim % (2 * self.heads) or self.layers < 1:
            raise ValueError("Query head dimensions must be positive and even")
        if self.ffn_dim != default_ffn_dim(self.dim):
            raise ValueError("SwiGLU width must follow 8/3 dim rounded up to a multiple of 64")
        if not 0 <= self.dropout < 1 or not self.rope_base > 1:
            raise ValueError("Invalid dropout or RoPE base")

    @classmethod
    def load(cls, path: Path) -> MDDPretrainConfig:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or raw.pop("name", None) != "mdd_dpt_pretrain":
            raise ValueError("Expected mdd_dpt_pretrain model config")
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("Pretraining configuration must declare every field exactly")
        for name in ("stem_channels", "mixed_channels", "residual_blocks"):
            raw[name] = tuple(raw[name])
        return cls(**raw)
