"""Configuration objects for tennis DETR-style pose model v2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TennisDetrV2Config:
    """High level hyper-parameters for :class:`TennisDETR_v2`.

    v2では階層エンコーダ構造（intra → inter → temporal）と
    分離出力（canonical + root）を採用。
    """

    # Transformer dimensions
    D_model: int = 256
    dim_feedforward: int = 1024
    nheads: int = 8
    decoder_layers: int = 6
    dropout: float = 0.1

    # v2用階層エンコーダパラメータ
    intra_layers: int = 3  # プレーヤー内エンコーダ層数
    inter_layers: int = 3  # プレーヤー間エンコーダ層数
    temporal_layers: int = 3  # 時間エンコーダ層数

    # Tokens / queries
    num_joints: int = 20
    num_court_points: int = 20
    num_queries: int = 20  # maximum number of players to track

    # Positional embeddings
    max_cameras: int = 8
    max_frames: int = 32
