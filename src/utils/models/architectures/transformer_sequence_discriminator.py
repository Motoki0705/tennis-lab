"""Transformer discriminator for arbitrary masked feature sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from src.utils.models.components import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.utils.models.components.ffn_layers import (
    SUPPORTED_FFN_TYPES,
    FFNType,
)
from src.utils.models.transformer_utils import build_self_attn_mask

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


class TransformerSequenceDiscriminator(nn.Module):
    """Score real-vs-fake sequence tensors with a CLS-token transformer."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta: float,
        ffn_type: FFNType,
        max_seq_len: int,
        invalid_init_std: float,
        cls_init_std: float,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {ffn_dim}")

        head_dim = hidden_dim // num_heads
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.rope_dim = int(rope_dim)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_dropout = nn.Dropout(float(dropout))
        self.invalid_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * float(invalid_init_std)
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * float(cls_init_std)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=float(rope_theta),
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, 1)

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_seq_len + 1,
            base=float(rope_theta),
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        sequence: Tensor,
        *,
        padding_mask: Tensor,
    ) -> Tensor:
        """Score ``(B,T,F)`` sequences, with ``True`` marking padded tokens."""
        if not isinstance(sequence, Tensor):
            raise TypeError("sequence must be a torch.Tensor.")
        if sequence.ndim != 3:
            raise ValueError(
                f"sequence must have shape (B,T,F), got {tuple(sequence.shape)}."
            )
        batch_size, seq_len, feature_dim = sequence.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {feature_dim}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if not isinstance(padding_mask, Tensor):
            raise TypeError("padding_mask must be a torch.Tensor.")
        if padding_mask.dtype != torch.bool:
            raise TypeError(
                "padding_mask must have dtype torch.bool, "
                f"got {padding_mask.dtype}."
            )
        if padding_mask.ndim != 2:
            raise ValueError(
                "padding_mask must have shape (B,T), "
                f"got rank {padding_mask.ndim} and shape {tuple(padding_mask.shape)}."
            )
        if padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "padding_mask must have shape (B,T), "
                f"got {tuple(padding_mask.shape)} vs {(batch_size, seq_len)}."
            )
        if padding_mask.device != sequence.device:
            raise ValueError("padding_mask and sequence must be on the same device.")

        x = self.input_projection(sequence)
        invalid = self.invalid_token.expand(batch_size, seq_len, -1)
        token_valid = ~padding_mask
        x = torch.where(token_valid.unsqueeze(-1), x, invalid)
        x = self.input_dropout(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_valid = torch.ones(
            batch_size,
            1,
            device=padding_mask.device,
            dtype=torch.bool,
        )
        attention_valid = torch.cat((cls_valid, token_valid), dim=1)
        attention_mask, _ = build_self_attn_mask(attention_valid)

        freqs_cis = self.get_buffer("freqs_cis")[: seq_len + 1]

        for block in self.blocks:
            x = cast(
                Tensor,
                block(x, freqs_cis=freqs_cis, attn_mask=attention_mask),
            )

        x = cast(Tensor, self.final_norm(x))
        logits = cast(Tensor, self.head(x[:, 0, :]))
        return logits.squeeze(-1)


def build_trajectory_discriminator(
    *,
    input_dim: int,
    disc_cfg: Mapping[str, Any],
) -> TransformerSequenceDiscriminator:
    """Build a :class:`TransformerSequenceDiscriminator` from a GAN config.

    Replicates the kwarg parsing performed by the task-specific discriminator
    wrappers (ball detection / BLCS).

    Args:
        input_dim: Feature dimension of the scored sequences (e.g. 2 for
            image-space ball trajectories, 3 for 3D trajectories).
        disc_cfg: The ``training.gan.discriminator`` config mapping.
    Returns:
        TransformerSequenceDiscriminator: The configured discriminator.
    """
    if type(input_dim) is not int:
        raise TypeError("input_dim must be exactly int.")
    expected = {
        "hidden_dim",
        "num_layers",
        "num_heads",
        "ffn_dim",
        "dropout",
        "rope_dim",
        "rope_theta",
        "ffn_type",
        "max_seq_len",
        "invalid_init_std",
        "cls_init_std",
    }
    missing = expected - set(disc_cfg)
    unknown = set(disc_cfg) - expected
    if missing or unknown:
        raise ValueError(
            "Invalid trajectory discriminator keys: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for key in (
        "hidden_dim",
        "num_layers",
        "num_heads",
        "ffn_dim",
        "rope_dim",
        "max_seq_len",
    ):
        if type(disc_cfg[key]) is not int:
            raise TypeError(f"trajectory discriminator {key} must be exactly int.")
    for key in ("dropout", "rope_theta", "invalid_init_std", "cls_init_std"):
        if type(disc_cfg[key]) is not float:
            raise TypeError(f"trajectory discriminator {key} must be exactly float.")
    if type(disc_cfg["ffn_type"]) is not str:
        raise TypeError("trajectory discriminator ffn_type must be exactly str.")
    if disc_cfg["ffn_type"] not in SUPPORTED_FFN_TYPES:
        raise ValueError(
            "trajectory discriminator ffn_type must be one of "
            f"{sorted(SUPPORTED_FFN_TYPES)!r}."
        )
    return TransformerSequenceDiscriminator(
        input_dim=input_dim,
        hidden_dim=disc_cfg["hidden_dim"],
        num_layers=disc_cfg["num_layers"],
        num_heads=disc_cfg["num_heads"],
        ffn_dim=disc_cfg["ffn_dim"],
        dropout=disc_cfg["dropout"],
        rope_dim=disc_cfg["rope_dim"],
        rope_theta=disc_cfg["rope_theta"],
        ffn_type=cast(FFNType, disc_cfg["ffn_type"]),
        max_seq_len=disc_cfg["max_seq_len"],
        invalid_init_std=disc_cfg["invalid_init_std"],
        cls_init_std=disc_cfg["cls_init_std"],
    )
