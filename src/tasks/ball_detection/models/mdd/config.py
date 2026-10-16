from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.models.components.ffn_layers import default_ffn_dim


@dataclass(frozen=True)
class MDDPoseConfig:
    compression: str
    pose_pooling: str | None
    readout: str
    frames: int
    stem_channels: tuple[int, int, int, int]
    mixed_channels: tuple[int, int]
    dim: int
    heads: int
    layers: int
    dropout: float
    rope_base: float

    @classmethod
    def load(cls, path: Path) -> MDDPoseConfig:
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        if raw.pop("name", None) != "mdd_pose":
            raise ValueError("Expected the MDD+pose coordinate model configuration")
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("MDD+pose config requires every field and no unknown fields")
        raw["stem_channels"] = tuple(raw["stem_channels"])
        raw["mixed_channels"] = tuple(raw["mixed_channels"])
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.compression not in {"conv2d", "average", "unshuffle", "haar"}:
            raise ValueError("Unknown MDD compression")
        if self.readout not in {"query", "pose", "query_only"}:
            raise ValueError("Unknown coordinate readout")
        if self.readout == "query_only":
            if self.pose_pooling is not None:
                raise ValueError("query_only requires pose_pooling: null; no pose branch is constructed")
        elif self.pose_pooling not in {"deepsets", "attention", "hierarchical", "gnn"}:
            raise ValueError("Pose-conditioned readouts require a known pose pooling")
        if self.frames != 32:
            raise ValueError("This review architecture uses 32 real frames")
        if len(self.stem_channels) != 4 or len(self.mixed_channels) != 2:
            raise ValueError("Require four spatial-stem widths and two mixed-block widths")
        if any(type(v) is not int or v < 1 for v in (*self.stem_channels, *self.mixed_channels)):
            raise ValueError("Encoder widths must be positive integers")
        if self.heads < 1 or self.dim < 4 or self.dim % (2 * self.heads) or self.layers < 1:
            raise ValueError("Attention head dimensions must be positive and even")
        if not 0 <= self.dropout < 1 or not self.rope_base > 1:
            raise ValueError("Invalid dropout or RoPE base")

    @property
    def requires_pose(self) -> bool:
        return self.readout != "query_only"

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
