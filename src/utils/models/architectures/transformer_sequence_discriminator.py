"""Transformer discriminator for arbitrary masked feature sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from src.utils.models.components import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


class TransformerSequenceDiscriminator(nn.Module):
    """Score real-vs-fake sequence tensors with a CLS-token transformer."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        ffn_type: str = "swiglu",
        max_seq_len: int = 120,
        invalid_init_std: float = 0.02,
        cls_init_std: float = 0.02,
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

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")
        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.rope_dim = int(rope_dim)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_dropout = nn.Dropout(float(dropout))
        self.invalid_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * float(invalid_init_std)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * float(cls_init_std))
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
                        rope_base=float(rope_theta),
                        ffn_type=cast(Literal["swiglu", "mlp"], ffn_type),
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

    def forward(self, sequence: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Score sequence tensors as real or fake."""
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(1)
        if sequence.ndim != 3:
            raise ValueError(
                "sequence must be (B, F) or (B, T, F), "
                f"got shape {tuple(sequence.shape)}"
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

        if mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=sequence.device, dtype=torch.bool)
        else:
            seq_mask = mask > 0
            if seq_mask.ndim == 1:
                seq_mask = seq_mask.unsqueeze(1)
            if seq_mask.ndim != 2:
                raise ValueError(
                    "mask must be (B,) or (B, T), "
                    f"got shape {tuple(seq_mask.shape)}"
                )
            if seq_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "mask shape must match sequence length, "
                    f"got {tuple(seq_mask.shape)} vs {(batch_size, seq_len)}"
                )

        x = self.input_projection(sequence)
        invalid = self.invalid_token.expand(batch_size, seq_len, -1)
        x = torch.where(seq_mask.unsqueeze(-1), x, invalid)
        x = self.input_dropout(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        freqs_cis = cast(Tensor, self.freqs_cis[: seq_len + 1])
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        cls_valid = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
        attn_valid = torch.cat([cls_valid, seq_mask], dim=1)
        attn_mask = attn_valid[:, None, :].expand(batch_size, seq_len + 1, seq_len + 1)

        for block in self.blocks:
            x = cast(Tensor, block(x, freqs_cis=freqs_cis, attn_mask=attn_mask))

        x = cast(Tensor, self.final_norm(x))
        logits = cast(Tensor, self.head(x[:, 0, :]))
        return logits.squeeze(-1)


def build_trajectory_discriminator(
    *,
    input_dim: int,
    disc_cfg: Mapping[str, Any],
    default_max_seq_len: int,
) -> TransformerSequenceDiscriminator:
    """Build a :class:`TransformerSequenceDiscriminator` from a GAN config.

    Replicates the kwarg parsing performed by the task-specific discriminator
    wrappers (ball detection / BLCS), including the ``invisible_init_std`` to
    ``invalid_init_std`` rename.

    Args:
        input_dim: Feature dimension of the scored sequences (e.g. 2 for
            image-space ball trajectories, 3 for 3D trajectories).
        disc_cfg: The ``training.gan.discriminator`` config mapping.
        default_max_seq_len: Fallback for ``max_seq_len`` when the config does
            not provide one (e.g. ``model.num_frames`` for ball detection or
            ``data.max_seq_len`` for BLCS).

    Returns:
        TransformerSequenceDiscriminator: The configured discriminator.
    """
    return TransformerSequenceDiscriminator(
        input_dim=int(input_dim),
        hidden_dim=int(disc_cfg.get("hidden_dim", 128)),
        num_layers=int(disc_cfg.get("num_layers", 4)),
        num_heads=int(disc_cfg.get("num_heads", 4)),
        ffn_dim=disc_cfg.get("ffn_dim", None),
        dropout=float(disc_cfg.get("dropout", 0.1)),
        rope_dim=disc_cfg.get("rope_dim", None),
        rope_theta=float(disc_cfg.get("rope_theta", 10000.0)),
        ffn_type=str(disc_cfg.get("ffn_type", "swiglu")),
        max_seq_len=int(disc_cfg.get("max_seq_len", default_max_seq_len)),
        invalid_init_std=float(disc_cfg.get("invisible_init_std", 0.02)),
        cls_init_std=float(disc_cfg.get("cls_init_std", 0.02)),
    )