"""Dataclass configuration for the tennis_mono_locrot model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TennisMonoLocRotConfig:
    """Configuration for the monocular location+rotation model.

    This model predicts only the root translation and yaw-like rotation of
    players in court coordinates from monocular 2D keypoints and court
    keypoints. There is no temporal dimension and no full 3D pose output.
    """

    # Feature dimensions
    D_model: int = 256
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Transformer layers
    nheads: int = 8
    intra_layers: int = 3
    inter_layers: int = 3
    decoder_layers: int = 3

    # Geometry
    num_joints: int = 17
    num_court_points: int = 20

    # Dataset-related limits (kept in sync with dataset config)
    max_cameras: int = 4
    max_players: int = 20
