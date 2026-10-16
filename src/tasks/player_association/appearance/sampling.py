"""Which frames of a camera-local track are used for appearance, and their crops.

Only actually observed boxes are candidates. A box is rejected, with its
reason counted, when it is small, touches the image border (truncated body) or
overlaps another observed box of the same camera (occlusion or a merged box).
The remaining frames are sampled evenly over the track.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.parts import NativeParts
from src.utils.geometry.bbox import pairwise_iou
from src.utils.video import OpenCVVideoFrameReader


@dataclass(frozen=True)
class CropSamplingConfig:
    min_height_px: float = 48.
    border_px: float = 4.
    max_overlap_iou: float = .1
    max_samples: int = 16

    def __post_init__(self) -> None:
        if self.min_height_px <= 0 or self.border_px < 0 or not 0 <= self.max_overlap_iou <= 1 or self.max_samples < 1:
            raise ValueError(f"Invalid crop sampling config: {self}")


@dataclass(frozen=True)
class TrackAppearance:
    """Embeddings of the sampled crops of one track; ``(0, 0)`` embeddings without samples."""

    frames: NDArray[np.int64]  # (K,)
    embeddings: NDArray[np.float32]  # (K, E)
    parts: NativeParts | None = None

    def __post_init__(self) -> None:
        if self.parts is not None and (len(self.parts.embeddings) != len(self.frames) or self.embeddings.shape != (len(self.frames), 0)):
            raise ValueError('Native track appearance must align frames and cannot have a whole-image proxy')


@dataclass(frozen=True)
class TrackSamples:
    frames: NDArray[np.int64]  # (K,) sampled frames, sorted
    rejected: dict[str, int] = field(default_factory=dict)  # reason -> observed frames rejected


def eligible_frames(boxes: NDArray[np.floating], observed: NDArray[np.bool_], image_size: tuple[int, int],
                    config: CropSamplingConfig) -> tuple[NDArray[np.bool_], list[Counter[str]]]:
    """``(D, T)`` eligibility of every observed box and per-track rejection counts."""
    tracks, frames = observed.shape
    if boxes.shape != (tracks, frames, 4):
        raise ValueError("Boxes must be (D, T, 4) aligned to observed (D, T)")
    width, height = image_size
    heights = boxes[..., 3] - boxes[..., 1]
    small = heights < config.min_height_px
    border = ((boxes[..., 0] < config.border_px) | (boxes[..., 1] < config.border_px)
              | (boxes[..., 2] > width - config.border_px) | (boxes[..., 3] > height - config.border_px))
    overlapped = np.zeros((tracks, frames), bool)
    for frame in np.flatnonzero(observed.sum(0) > 1):
        rows = np.flatnonzero(observed[:, frame])
        iou = pairwise_iou(boxes[rows, frame], boxes[rows, frame])
        np.fill_diagonal(iou, 0)
        overlapped[rows, frame] = iou.max(1) > config.max_overlap_iou
    eligible = observed & ~small & ~border & ~overlapped
    reasons: list[Counter[str]] = []
    for row in range(tracks):
        counts: Counter[str] = Counter()
        seen = observed[row]
        counts["small"] = int((seen & small[row]).sum())
        counts["truncated"] = int((seen & ~small[row] & border[row]).sum())
        counts["overlapped"] = int((seen & ~small[row] & ~border[row] & overlapped[row]).sum())
        reasons.append(+counts)
    return eligible, reasons


def sample_tracks(boxes: NDArray[np.floating], observed: NDArray[np.bool_], image_size: tuple[int, int],
                  config: CropSamplingConfig) -> list[TrackSamples]:
    eligible, reasons = eligible_frames(boxes, observed, image_size, config)
    samples = []
    for row in range(len(observed)):
        frames = np.flatnonzero(eligible[row])
        if len(frames) > config.max_samples:
            frames = frames[np.unique(np.linspace(0, len(frames) - 1, config.max_samples).round().astype(np.int64))]
        samples.append(TrackSamples(frames.astype(np.int64), dict(reasons[row])))
    return samples


def crop(frame_bgr: NDArray[np.uint8], box: NDArray[np.floating], size: tuple[int, int]) -> NDArray[np.float32]:
    """RGB ``(3, H, W)`` in ``[0, 1]`` of ``box`` resized (not letterboxed) to ``size = (H, W)``, as Re-ID models are trained."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = np.round(box).astype(int)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Crop box {box} is empty inside a {width}x{height} frame")
    patch = cv2.resize(frame_bgr[y1:y2, x1:x2], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(patch[..., ::-1].transpose(2, 0, 1), np.float32) / 255.


def embed_samples(video: Path, boxes: NDArray[np.floating], samples: list[TrackSamples], encoders: Iterable[AppearanceEncoder],
                  *, batch_size: int = 64) -> dict[str, list[NDArray[np.float32]]]:
    """Per encoder, per track ``(K, D)`` embeddings of the sampled frames (one pass over the video)."""
    encoders = list(encoders)
    wanted: dict[int, list[int]] = {}
    for row, track in enumerate(samples):
        for frame in track.frames.tolist():
            wanted.setdefault(frame, []).append(row)
    sizes = {encoder.input_size for encoder in encoders}
    crops: dict[tuple[int, int], list[tuple[int, NDArray[np.float32]]]] = {size: [] for size in sizes}
    if wanted:
        for packet in OpenCVVideoFrameReader(video, max_frames=max(wanted) + 1):
            for row in wanted.get(packet.index, ()):
                for size in sizes:
                    crops[size].append((row, crop(packet.frame, boxes[row, packet.index], size)))
        missing = sum(map(len, wanted.values())) - len(next(iter(crops.values())))
        if missing:
            raise RuntimeError(f"{video} ended before {missing} sampled crops were read")
    result: dict[str, list[NDArray[np.float32]]] = {}
    for encoder in encoders:
        items = crops[encoder.input_size]
        rows = np.asarray([row for row, _ in items], np.int64)
        embedded = [encoder.embed(torch.from_numpy(np.stack([image for _, image in items[start:start + batch_size]]))).numpy()
                    for start in range(0, len(items), batch_size)]
        matrix = np.concatenate(embedded) if embedded else np.zeros((0, 0), np.float32)
        result[encoder.name] = [matrix[rows == row] for row in range(len(samples))]
    return result


def embed_tracks(video: Path, boxes: NDArray[np.floating], observed: NDArray[np.bool_], image_size: tuple[int, int],
                 encoder: AppearanceEncoder, config: CropSamplingConfig) -> tuple[tuple[TrackAppearance, ...], list[TrackSamples]]:
    """Appearance of every track of one camera and the samples it was built from (with rejection counts)."""
    samples = sample_tracks(boxes, observed, image_size, config)
    embedded = embed_samples(video, boxes, samples, [encoder])[encoder.name]
    appearances = tuple(TrackAppearance(track.frames, embedded[row].astype(np.float32) if len(track.frames) else np.zeros((0, 0), np.float32))
                        for row, track in enumerate(samples))
    return appearances, samples
