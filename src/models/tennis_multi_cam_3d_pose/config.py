"""Configuration objects for tennis DETR-style pose model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TennisDetrConfig:
    """High level hyper-parameters for :class:`TennisDETR`.

    This config intentionally keeps the model generic with respect to the
    number of cameras and frames. Values such as ``max_cameras`` and
    ``max_frames`` specify upper bounds for embedding tables; the actual
    shapes can be smaller at runtime.
    """

    # Transformer dimensions
    D_model: int = 256
    dim_feedforward: int = 1024
    nheads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout: float = 0.1

    # Tokens / queries
    num_joints: int = 20
    num_court_points: int = 20
    num_queries: int = 20  # maximum number of players to track

    # Positional embeddings
    max_cameras: int = 8
    max_frames: int = 32
