"""Configuration objects for tennis DETR-style pose model v3.

v3 shares most hyper-parameters with v2, but is intended for the
track-aware scene transformer architecture.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TennisDetrV3Config:
    """High level hyper-parameters for :class:`TennisDETR_v3`.

    The v3 model uses a hierarchical encoder over detections and a
    track-aware temporal decoder over per-player query tracks.
    """

    # Transformer dimensions
    D_model: int = 256
    dim_feedforward: int = 1024
    nheads: int = 8
    decoder_layers: int = 6
    dropout: float = 0.1

    # Hierarchical encoder parameters
    intra_layers: int = 3
    inter_layers: int = 3
    temporal_layers: int = 3  # used for the track-temporal encoder

    # Tokens / queries
    num_joints: int = 20
    num_court_points: int = 20
    num_queries: int = 20  # maximum number of players to track

    # Positional embeddings
    max_cameras: int = 8
    max_frames: int = 32
