"""Type definitions for event detection datasets."""

from __future__ import annotations

from typing import TypedDict

import torch


class EventUVSample(TypedDict):
    """Single-sample schema for UV-based event detection."""

    ball_uv: torch.Tensor  # (T, 2)
    ball_vis: torch.Tensor  # (T,)
    court_kp: torch.Tensor  # (20, 2)
    court_vis: torch.Tensor  # (20,)
    targets: torch.Tensor  # (T, E)
    seq_len: torch.Tensor  # scalar (int)


class Event3DSample(TypedDict):
    """Single-sample schema for 3D-trajectory-based event detection."""

    ball_pos_world: torch.Tensor  # (T, 3), meters
    targets: torch.Tensor  # (T, E)
    seq_len: torch.Tensor  # scalar (int)


class EventUVBatch(TypedDict):
    """Batched schema for UV-based event detection."""

    ball_uv: torch.Tensor  # (B, T, 2)
    ball_vis: torch.Tensor  # (B, T)
    ball_mask: torch.Tensor  # (B, T)
    court_kp: torch.Tensor  # (B, 20, 2)
    court_vis: torch.Tensor  # (B, 20)
    targets: torch.Tensor  # (B, T, E)
    seq_len: torch.Tensor  # (B,)


class Event3DBatch(TypedDict):
    """Batched schema for 3D-trajectory-based event detection."""

    ball_pos_world: torch.Tensor  # (B, T, 3)
    targets: torch.Tensor  # (B, T, E)
    seq_len: torch.Tensor  # (B,)
