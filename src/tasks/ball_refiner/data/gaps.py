"""Label-independent evidence gaps and camera-grouped validation partition."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import ClipRecord
from src.tasks.ball_detection.model_io.contracts import BallCandidates
from src.tasks.ball_refiner.data.windows import CANDIDATE_FIELDS
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput


def validation_partition(clips: tuple[ClipRecord, ...], seed: int) -> dict[str, list[str]]:
    """Keep all cameras of a Meiji temporal clip together, independent of labels."""
    groups: dict[str, list[str]] = {}
    for clip in clips:
        parts = clip.clip_id.split("/")
        if clip.source != "meiji" or clip.split != "val" or len(parts) != 4 or not parts[-1].startswith("cam"):
            raise ValueError("Selection/calibration partition requires Meiji val clip/camera IDs")
        key = "/".join(parts[:-1])
        if key not in groups:
            groups[key] = []
        groups[key].append(clip.clip_id)
    if len(groups) < 2:
        raise ValueError("Need at least two temporal clips to separate selection and calibration")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    return {
        "selection": sorted(clip for key in ordered[::2] for clip in groups[key]),
        "calibration": sorted(clip for key in ordered[1::2] for clip in groups[key]),
    }


def fixed_gap_mask(
    frames: int, *, clip_id: str, block_length: int, lengths: tuple[int, ...], seed: int,
) -> NDArray[np.bool_]:
    """Centre gaps in global blocks, cycling lengths; never consult target UV/masks."""
    if not lengths or min(lengths) < 1 or max(lengths) >= block_length or frames < 1:
        raise ValueError("Gap lengths must be positive and shorter than the block")
    offset = int.from_bytes(hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()[:8], "little")
    mask: NDArray[np.bool_] = np.zeros(frames, dtype=np.bool_)
    for index, block in enumerate(range(0, frames, block_length)):
        length = lengths[(index + offset) % len(lengths)]
        available = min(block_length, frames - block)
        if available < length:
            continue  # explicit incomplete-tail policy; do not shorten a gap
        start = block + (available - length) // 2
        mask[start:start + length] = True
    return mask


def random_gap_mask(
    batch: int, frames: int, *, lengths: tuple[int, ...], probability: float, generator: torch.Generator,
) -> torch.Tensor:
    if not lengths or min(lengths) < 1 or max(lengths) >= frames or not 0 <= probability <= 1:
        raise ValueError("Invalid train evidence gap policy")
    mask = torch.zeros(batch, frames, dtype=torch.bool)
    for row in range(batch):
        if float(torch.rand((), generator=generator)) < probability:
            length = lengths[int(torch.randint(len(lengths), (), generator=generator))]
            start = int(torch.randint(frames - length + 1, (), generator=generator))
            mask[row, start:start + length] = True
    return mask


def mask_detector_evidence(inputs: Refiner2DInput, mask: torch.Tensor) -> Refiner2DInput:
    """Remove every candidate field without modifying the cached source tensors."""
    if mask.shape != inputs.timestamps_seconds.shape or mask.dtype != torch.bool:
        raise ValueError("Evidence gap mask must be bool B,T")
    if mask.device != inputs.candidates.coords.device:
        raise ValueError("Gap and candidates must share a device")
    candidates = BallCandidates(**{
        name: getattr(inputs.candidates, name).masked_fill(
            mask.reshape(*mask.shape, *((1,) * (getattr(inputs.candidates, name).ndim - 2))), 0,
        ) for name in CANDIDATE_FIELDS
    }, config=inputs.candidates.config)
    return replace(inputs, candidates=candidates)
