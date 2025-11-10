"""Collate utilities for variable length tracking batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from src.datasets.dancetrack import TargetFrame, TrackingSample


@dataclass(slots=True)
class SceneBatch:
    """Batched representation consumed by the LightningModule."""

    frames: Tensor
    targets: list[list[TargetFrame]]
    padding_mask: Tensor
    sequence_ids: list[str]


def collate_tracking(samples: Sequence[TrackingSample]) -> SceneBatch:
    """Pad variable length samples into a dense `SceneBatch`."""

    if not samples:
        msg = "collate_tracking received no samples"
        raise ValueError(msg)
    max_len = max(sample.frames.shape[0] for sample in samples)
    batch = samples[0].frames
    B = len(samples)
    frames = batch.new_zeros((B, max_len, batch.shape[1], batch.shape[2], batch.shape[3]))
    padding_mask = torch.ones((B, max_len), dtype=torch.bool, device=batch.device)
    batched_targets: list[list[TargetFrame]] = []
    sequence_ids: list[str] = []
    for idx, sample in enumerate(samples):
        T = sample.frames.shape[0]
        frames[idx, :T] = sample.frames
        padding_mask[idx, :T] = False
        padded_targets = list(sample.targets)
        if T < max_len:
            device = sample.frames.device
            padded_targets.extend(TargetFrame.empty(device=device) for _ in range(max_len - T))
        batched_targets.append(padded_targets)
        sequence_ids.append(sample.sequence_id)
    return SceneBatch(frames=frames, targets=batched_targets, padding_mask=padding_mask, sequence_ids=sequence_ids)
