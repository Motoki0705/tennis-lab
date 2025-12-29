"""Type definitions for WASB data structures.

This module provides TypedDict schemas for dataset batches, complementing
the existing dataclasses (TennisLabelRow, SequenceSample) defined elsewhere.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class BallDetectionSample(TypedDict):
    """Schema for ball detection dataset sample.

    Used by BallDetectionSequenceDataset.__getitem__(). Contains a sliding
    window of frames with target annotations for the last frames_out frames.
    """

    frames: torch.Tensor  # (T, C, H, W) input frame sequence
    targets_px: torch.Tensor  # (frames_out, 2) target positions in pixels
    targets_norm: torch.Tensor  # (frames_out, 2) target positions normalized [0, 1]
    target_heatmaps: torch.Tensor  # (frames_out, H, W) Gaussian heatmaps
    visibility: torch.Tensor  # (frames_out,) visibility flags (0/1/2)
    scores: torch.Tensor  # (frames_out,) confidence scores
    match: str  # match/game directory name
    clip: str  # clip directory name
    frame_paths: list[str]  # paths to frame images


class TrajectoryWindowSample(TypedDict):
    """Schema for trajectory window dataset sample.

    Used by TrajectoryWindowDataset for ball trajectory prediction tasks.
    """

    frames: torch.Tensor  # (T, C, H, W) input frames
    ball_positions: torch.Tensor  # (T, 2) ball positions over time
    visibility: torch.Tensor  # (T,) visibility mask
    match: str
    clip: str


class EventDetectionSample(TypedDict):
    """Schema for event detection dataset sample.

    Used by EventDetectionDataset for detecting bounce/net events.
    """

    frames: torch.Tensor  # (T, C, H, W) input frames
    event_labels: torch.Tensor  # (num_event_types,) binary event occurrence
    match: str
    clip: str


class PatchEmbeddingSample(TypedDict):
    """Schema for patch embedding dataset sample.

    Used by PatchEmbeddingsDataset for DinoV3 patch token features.
    """

    patch_tokens: torch.Tensor  # (T, num_patches, embed_dim) patch embeddings
    targets: torch.Tensor  # (frames_out, 2) target ball positions
    visibility: torch.Tensor  # (frames_out,) visibility flags
    match: str
    clip: str
