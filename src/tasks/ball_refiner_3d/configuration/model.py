"""Generator and discriminator architecture contracts."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES


@dataclass(frozen=True)
class ModelConfig:
    dimensions: int
    architecture: str
    width: int
    layers: int
    heads: int
    dropout: float
    flow_steps: int
    ffn_dim: int
    rope_dim: int
    rope_theta: float
    ffn_type: str
    # Field (wind, k_drag, k_magnus, surface) and segment initial-state heads.
    physics_heads: bool

    def __post_init__(self) -> None:
        if self.dimensions != 3 or self.architecture not in ("regression", "flow"):
            raise ValueError("Require dimensions=3 and architecture=regression|flow")
        if self.physics_heads and self.architecture != "regression":
            raise ValueError("Physics heads are only supported by direct regression")
        if (
            self.width < 8
            or self.heads < 1
            or self.width % self.heads
            or self.width % 2
        ):
            raise ValueError("width must be even and divisible by heads")
        if min(self.layers, self.flow_steps) < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid temporal model configuration")
        if (
            self.ffn_dim < 1
            or self.rope_dim < 2
            or self.rope_dim % 2
            or self.rope_dim > self.width // self.heads
            or self.rope_theta <= 0
        ):
            raise ValueError("Invalid FFN or RoPE dimensions")
        if self.ffn_type not in SUPPORTED_FFN_TYPES:
            raise ValueError(f"Unsupported FFN type: {self.ffn_type}")


@dataclass(frozen=True)
class DiscriminatorConfig:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    rope_dim: int
    rope_theta: float
    ffn_type: str
    max_seq_len: int
    invalid_init_std: float
    cls_init_std: float

    def __post_init__(self) -> None:
        if (
            self.name != "trajectory_transformer"
            or self.ffn_type not in SUPPORTED_FFN_TYPES
        ):
            raise ValueError("Require trajectory_transformer with a supported FFN type")
        if (
            min(
                self.hidden_dim,
                self.num_heads,
                self.num_layers,
                self.ffn_dim,
                self.max_seq_len,
            )
            < 1
            or self.hidden_dim % self.num_heads
        ):
            raise ValueError("Invalid discriminator dimensions")
        if (
            self.rope_dim < 2
            or self.rope_dim % 2
            or self.rope_dim > self.hidden_dim // self.num_heads
            or self.rope_theta <= 0
        ):
            raise ValueError("Invalid discriminator RoPE configuration")
        if (
            not 0 <= self.dropout < 1
            or min(self.invalid_init_std, self.cls_init_std) < 0
        ):
            raise ValueError("Invalid discriminator dropout or initialization")
