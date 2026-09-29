"""Bounded-memory, frame-indexed person crops shared by pose and image features."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.submodules.vendor.gvhmr.hmr2.preproc import (
    IMAGE_MEAN,
    IMAGE_STD,
    crop_and_resize,
)
from src.utils.video import OpenCVVideoFrameReader


def iter_person_crops(
    video_path: str | Path, boxes_xys: Tensor, frame_indices: Tensor, *, batch_size: int
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield ImageNet crops and original-pixel boxes in requested frame order."""
    # Construct the reader lazily; an empty request must not open the video.
    frames = ((packet.index, packet.frame) for packet in OpenCVVideoFrameReader(Path(video_path)))
    yield from iter_frame_person_crops(frames, boxes_xys, frame_indices, batch_size=batch_size)


def iter_frame_person_crops(
    frames_bgr: Iterable[tuple[int, NDArray[np.uint8]]],
    boxes_xys: Tensor, frame_indices: Tensor, *, batch_size: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Crop an ordered BGR stream without encoding a surrogate video.

    Frame IDs must increase strictly, and every requested ID must occur.
    The request may repeat an ID for multiple people. Images must have a
    constant size and use the boxes' pixel coordinate system. Only frames
    through the last requested ID are consumed; the caller owns full-clip
    provenance. Crop/downsampling/normalization are shared with video input.
    """
    if boxes_xys.ndim != 2 or boxes_xys.shape[-1] != 3 or frame_indices.shape != (len(boxes_xys),):
        raise ValueError("Indexed crops require boxes (F,3) and frame indices (F,)")
    if boxes_xys.device.type != "cpu" or frame_indices.device.type != "cpu" or frame_indices.dtype != torch.int64:
        raise TypeError("Indexed crop requests require CPU tensors and int64 frame indices")
    if not boxes_xys.is_floating_point() or not bool(torch.isfinite(boxes_xys).all()) or bool((boxes_xys[:, 2] <= 0).any()):
        raise ValueError("Crop boxes must be finite with positive size")
    if type(batch_size) is not int or batch_size < 1 or bool((frame_indices < 0).any()) or bool((frame_indices.diff() < 0).any()):
        raise ValueError("Crop indices must be nonnegative and ordered")
    if not len(frame_indices):
        return
    indices = frame_indices.tolist()
    row = 0
    crops: list[np.ndarray] = []
    boxes: list[Tensor] = []
    previous = -1
    shape: tuple[int, ...] | None = None
    for index, frame in frames_bgr:
        if type(index) is not int or index <= previous:
            raise ValueError("Source frame IDs must be strictly increasing nonnegative integers")
        previous = index
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2:
            raise ValueError("Source frames must be uint8 BGR (H,W,3), with H,W >= 2")
        if shape is not None and frame.shape != shape:
            raise ValueError("Source frame size changed within the crop stream")
        shape = frame.shape
        if index > indices[row]:
            raise ValueError(f"Missing requested source frame {indices[row]}")
        if index < indices[row]:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        small = cv2.resize(rgb, (int(width * .5), int(height * .5)))
        while row < len(indices) and indices[row] == index:
            box = boxes_xys[row].float()
            factor = float(box[2]) * .5 / 256 / 2
            source = cv2.GaussianBlur(small, (5, 5), (factor - 1) / 2) if factor > 1.1 else small
            crop, _ = crop_and_resize(source, box[:2].numpy() * .5, float(box[2]) * .5, 256, enlarge_ratio=1.)
            crops.append(crop)
            boxes.append(box)
            row += 1
            if len(crops) == batch_size:
                pixels = torch.from_numpy(np.stack(crops))
                yield ((pixels / 255. - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2), torch.stack(boxes)
                crops, boxes = [], []
        if row == len(indices):
            break
    if row != len(indices):
        raise ValueError(f"Frame stream ended before requested source frame {indices[row]}")
    if crops:
        pixels = torch.from_numpy(np.stack(crops))
        yield ((pixels / 255. - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2), torch.stack(boxes)
