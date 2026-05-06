"""Transformer discriminator for PLCS pose sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.models.architectures import TransformerSequenceDiscriminator

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSPoseSequenceDiscriminator(TransformerSequenceDiscriminator):
    """Discriminator over concatenated PLCS position and rotation outputs."""

    @classmethod
    def from_config(cls, config: DictConfig) -> "PLCSPoseSequenceDiscriminator":
        train_cfg = config.get("training", {}) or {}
        gan_cfg = train_cfg.get("gan", {}) or {}
        disc_cfg = gan_cfg.get("discriminator", {}) or {}
        model_cfg = config.get("model", {}) or {}

        return cls(
            input_dim=5,
            hidden_dim=int(disc_cfg.get("hidden_dim", 128)),
            num_layers=int(disc_cfg.get("num_layers", 4)),
            num_heads=int(disc_cfg.get("num_heads", 4)),
            ffn_dim=disc_cfg.get("ffn_dim", None),
            dropout=float(disc_cfg.get("dropout", 0.1)),
            rope_dim=disc_cfg.get("rope_dim", None),
            rope_theta=float(disc_cfg.get("rope_theta", 10000.0)),
            ffn_type=str(disc_cfg.get("ffn_type", "swiglu")),
            max_seq_len=int(disc_cfg.get("max_seq_len", model_cfg.get("max_seq_len", 120))),
            invalid_init_std=float(disc_cfg.get("invalid_init_std", 0.02)),
            cls_init_std=float(disc_cfg.get("cls_init_std", 0.02)),
        )