"""Incremental RGB inference with deterministic, score-independent overlap choice."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.evaluation.holdout_inference import HoldoutPredictor
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallCandidates,
)
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.utils.video import (
    BgrToTensorTransform,
    FramePacket,
    iter_temporal_batches,
    iter_temporal_windows,
)

WINDOW_SELECTION = "nearest_window_centre_then_earlier_start"


def infer_clip_evidence(
    store: BallFrameStore, clip: ClipRecord, predictor: HoldoutPredictor, *,
    image_size_hw: tuple[int, int], stride: int, batch_size: int, config: BallCandidateConfig,
) -> ClipEvidence:
    """Keep all candidate fields from the same selected detector window.

    The detector's own temporal window can exceed the later refiner window.
    Its length/stride and per-frame origin are explicit in the cache. No RGB
    padding, score selection, GT selection, candidate gate or temporal fill.
    """
    length, n = predictor.configured_frames, clip.frame_count
    if store.clip_by_id(clip.clip_id) != clip:
        raise ValueError("Clip does not belong to this store")
    if n < length:
        raise ValueError(f"{clip.clip_id}: short clip {n} < {length}; repeated RGB padding is forbidden")
    if not 1 <= stride <= length or batch_size < 1 or min(image_size_hw) <= 0:
        raise ValueError("Invalid detector stride, batch size or image size")
    if min(clip.width, clip.height, clip.source_width, clip.source_height) <= 1:
        raise ValueError("Image sizes must support endpoint normalization")
    transform = BgrToTensorTransform(image_size=image_size_hw, normalize_imagenet=False)
    stream = (
        FramePacket(index=i, frame=transform(store.read_bgr(store.row_of(clip, i))),
                    original_size=(clip.width, clip.height))
        for i in range(n)
    )
    windows = iter_temporal_windows(stream, sequence_length=length, stride=stride, tail_policy="backfill")
    k, p = config.max_candidates, config.patch_size
    candidates = BallCandidates(
        coords=torch.zeros(1, n, k, 2), scores=torch.zeros(1, n, k),
        valid=torch.zeros(1, n, k, dtype=torch.bool), cells=torch.zeros(1, n, k, 2, dtype=torch.int64),
        patches=torch.zeros(1, n, k, p, p), patch_valid=torch.zeros(1, n, k, p, p, dtype=torch.bool),
        config=config,
    )
    argmax_uv = np.zeros((n, 2), np.float32)
    argmax_score = np.zeros(n, np.float32)
    starts, slots = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    distances = np.full(n, np.iinfo(np.int64).max, np.int64)
    factor = torch.tensor([clip.width - 1, clip.height - 1], dtype=torch.float32)
    factor /= clip.scale * torch.tensor([clip.source_width - 1, clip.source_height - 1])
    heatmap_size: tuple[int, int] | None = None
    for batch in iter_temporal_batches(windows, batch_size=batch_size):
        prediction = predictor.predict(batch.tensor, candidate_config=config)
        if prediction.candidates.config != config:
            raise ValueError("Predictor changed the requested candidate contract")
        size = (int(prediction.heatmaps.shape[-2]), int(prediction.heatmaps.shape[-1]))
        if heatmap_size is not None and heatmap_size != size:
            raise ValueError("Native heatmap size changed within a clip")
        heatmap_size = size
        for b, window in enumerate(batch.windows):
            for t, frame in enumerate(window.frame_indices):
                distance = abs(2 * t - (length - 1))
                if distance >= distances[frame]:
                    continue  # ties retain the earlier start; windows are ordered
                distances[frame] = distance
                starts[frame], slots[frame] = window.start_index, t
                argmax_uv[frame] = (prediction.coords[b, t] * factor).numpy()
                argmax_score[frame] = prediction.confidence[b, t].item()
                for name in ("coords", "scores", "valid", "cells", "patches", "patch_valid"):
                    value = getattr(prediction.candidates, name)[b, t]
                    getattr(candidates, name)[0, frame] = value * factor if name == "coords" else value
    if heatmap_size is None or (starts < 0).any():
        raise ValueError(f"{clip.clip_id}: incomplete detector evidence")
    rows = store.clip_rows(clip)
    pts = store.frames["pts"][rows].copy()
    seconds = ((pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
    return ClipEvidence(
        frame_index=store.frames["frame_index"][rows].copy(), pts=pts, timestamps_seconds=seconds,
        window_start=starts, time_index=slots, argmax_uv=argmax_uv, argmax_score=argmax_score,
        candidates=candidates, heatmap_size_hw=heatmap_size, window_length=length,
    )
