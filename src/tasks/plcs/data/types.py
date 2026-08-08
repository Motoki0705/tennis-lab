"""Type definitions for PLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the PLCS pipeline.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class PLCSBatch(TypedDict):
    """Unified PLCS batch schema for frame/sequence/single/multiview modes.

    Shapes use camera-time ordering:
    - ``human_kp``: (B, N, T, 17, 2)
    - ``court_kp``: (B, N, T, 20, 2)
    - ``human_vis``: (B, N, T, 17)
    - ``human_mask``: (B, N, T), padding mask (True/1=valid token)
    - ``court_vis``: (B, N, T, 20)
    - ``position``: (B, T, 3)
    - ``rotation``: (B, T, 2)
    """

    human_kp: torch.Tensor
    court_kp: torch.Tensor
    human_vis: torch.Tensor
    human_mask: torch.Tensor
    court_vis: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
