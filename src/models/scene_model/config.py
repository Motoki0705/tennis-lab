"""Configuration objects for the scene model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SceneConfig:
    """High level hyper-parameters for :class:`SceneModel`."""

    # Image / ViT
    img_size: int
    patch_size: int
    vit_depth: int
    vit_heads: int
    D_model: int

    # Temporal encoder (RoPE)
    temporal_depth: int
    temporal_heads: int
    max_T_hint: int

    # Time PE
    fps: float
    absK: int
    abs_Thorizon: float
    relQ: int
    rel_wmin: float
    rel_wmax: float

    # Decoder
    num_queries: int
    decoder_layers: int
    num_points: int
    window_k: int
    offset_mode: str = "per_tau"
    tbptt_detach: bool = True

    # Prediction head
    smpl_param_dim: int = 0
