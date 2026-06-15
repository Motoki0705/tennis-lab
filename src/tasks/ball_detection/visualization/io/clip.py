"""Clip IO helpers for ball-detection visualization.

Loads a clip directory (frames + ``Label.csv``) into the preprocessed tensors
consumed by inference and the original RGB frames used for rendering.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

from src.tasks.ball_detection.data.augmentation import normalize_tensor_images_imagenet
from src.tasks.ball_detection.data.types import FrameLabel
from src.tasks.base.visualization.frames import read_rgb, resolve_image_paths


@dataclass(frozen=True)
class ClipSequence:
    """Loaded and preprocessed clip data for inference and rendering."""

    frame_names: tuple[str, ...]
    render_frames_rgb: tuple[torch.Tensor, ...]
    model_images: torch.Tensor
    gt_coords_px: torch.Tensor
    gt_visibility: torch.Tensor


def load_clip_sequence(
    *,
    clip_dir: Path,
    sequence_length: int,
    image_size_hw: tuple[int, int],
    max_frames: int | None,
    normalize_imagenet: bool,
    imagenet_mean: tuple[float, float, float],
    imagenet_std: tuple[float, float, float],
) -> ClipSequence:
    """Load and preprocess a clip directory for visualization.

    Args:
        clip_dir: Directory holding ``*.jpg`` frames and ``Label.csv``.
        sequence_length: Model temporal window length (``model.num_frames``).
        image_size_hw: Target ``(height, width)`` for model/render frames.
        max_frames: Optional cap on the number of frames.
        normalize_imagenet: Whether to ImageNet-normalize model images.
        imagenet_mean: ImageNet mean (used when normalizing).
        imagenet_std: ImageNet std (used when normalizing).

    Returns:
        A :class:`ClipSequence` with render frames, model images and GT labels.
    """
    label_path = clip_dir / "Label.csv"
    if not clip_dir.exists():
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label.csv not found for clip: {clip_dir}")

    frame_paths = resolve_image_paths(clip_dir, max_frames=max_frames)
    if len(frame_paths) < sequence_length:
        raise ValueError(
            f"Clip {clip_dir} has only {len(frame_paths)} frames, "
            f"but model.num_frames={sequence_length}."
        )

    labels = _read_label_csv(label_path)
    image_height, image_width = image_size_hw
    render_frames_rgb: list[torch.Tensor] = []
    model_frames: list[torch.Tensor] = []
    gt_coords_px: list[tuple[float, float]] = []
    gt_visibility: list[float] = []
    frame_names: list[str] = []

    for frame_path in frame_paths:
        frame_rgb = read_rgb(frame_path)
        original_height, original_width = frame_rgb.shape[:2]
        resized_rgb = cv2.resize(
            frame_rgb,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        render_frames_rgb.append(torch.from_numpy(resized_rgb.copy()).to(torch.uint8))

        model_frame = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0
        model_frames.append(model_frame)

        label = labels.get(frame_path.name, FrameLabel(visibility=0.0, x=0.0, y=0.0))
        if label.visibility > 0:
            gt_coords_px.append(
                (
                    label.x * image_width / max(original_width, 1),
                    label.y * image_height / max(original_height, 1),
                )
            )
            gt_visibility.append(1.0)
        else:
            gt_coords_px.append((0.0, 0.0))
            gt_visibility.append(0.0)

        frame_names.append(frame_path.name)

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
        gt_coords_px=torch.tensor(gt_coords_px, dtype=torch.float32),
        gt_visibility=torch.tensor(gt_visibility, dtype=torch.float32),
    )


def _read_label_csv(path: Path) -> dict[str, FrameLabel]:
    labels: dict[str, FrameLabel] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"file name", "visibility", "x-coordinate", "y-coordinate"}
        missing = required_fields.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns in {path}: {sorted(missing)}")
        for row in reader:
            frame_name = str(row["file name"]).strip()
            labels[frame_name] = FrameLabel(
                visibility=float(row["visibility"] or 0.0),
                x=float(row["x-coordinate"] or 0.0),
                y=float(row["y-coordinate"] or 0.0),
            )
    return labels
