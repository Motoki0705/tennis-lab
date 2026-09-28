"""Clip IO helpers for ball-detection visualization.

Loads a clip from the versioned ball frame store into the preprocessed tensors
consumed by inference and the original RGB frames used for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

from src.tasks.ball_detection.data.components.augmentation import (
    normalize_tensor_images_imagenet,
)
from src.tasks.ball_detection.data.store import BallFrameStore


@dataclass(frozen=True)
class ClipSequence:
    """Loaded and preprocessed clip data for inference and rendering."""

    frame_names: tuple[str, ...]
    render_frames_rgb: tuple[torch.Tensor, ...]
    model_images: torch.Tensor


def load_clip_sequence(
    *,
    store_dir: Path,
    clip_id: str,
    sequence_length: int,
    image_size_hw: tuple[int, int],
    max_frames: int | None,
    normalize_imagenet: bool,
    imagenet_mean: tuple[float, float, float],
    imagenet_std: tuple[float, float, float],
) -> ClipSequence:
    """Load and preprocess a stored clip for visualization.

    Args:
        store_dir: Absolute path of a ball frame store version.
        clip_id: Exact clip_id from its metadata.
        sequence_length: Model temporal window length (``model.num_frames``).
        image_size_hw: Target ``(height, width)`` for model/render frames.
        max_frames: Optional cap on the number of frames.
        normalize_imagenet: Whether to ImageNet-normalize model images.
        imagenet_mean: ImageNet mean (used when normalizing).
        imagenet_std: ImageNet std (used when normalizing).

    Returns:
        A :class:`ClipSequence` with render frames, model images and GT labels.
    """
    store = BallFrameStore(store_dir)
    clip = store.clip_by_id(clip_id)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    rows = store.clip_rows(clip)
    if max_frames is not None:
        rows = rows[:max_frames]
    if len(rows) < sequence_length:
        raise ValueError(f"Clip {clip_id} has only {len(rows)} frames for window {sequence_length}")

    image_height, image_width = image_size_hw
    render_frames_rgb: list[torch.Tensor] = []
    model_frames: list[torch.Tensor] = []
    frame_names: list[str] = []

    for row in rows:
        frame_rgb = cv2.cvtColor(store.read_bgr(int(row)), cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(
            frame_rgb,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        render_frames_rgb.append(torch.from_numpy(resized_rgb.copy()).to(torch.uint8))

        model_frame = (
            torch.from_numpy(resized_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0
        )
        model_frames.append(model_frame)

        frame_names.append(store.frame_key(int(row)))

    model_images = torch.stack(model_frames, dim=0)
    if normalize_imagenet:
        model_images = normalize_tensor_images_imagenet(
            model_images,
            mean=imagenet_mean,
            std=imagenet_std,
        )

    return ClipSequence(
        frame_names=tuple(frame_names),
        render_frames_rgb=tuple(render_frames_rgb),
        model_images=model_images,
    )
