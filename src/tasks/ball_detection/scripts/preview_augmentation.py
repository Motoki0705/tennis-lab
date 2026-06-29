"""Generate contact sheets for ball-detection sequence augmentations.

Usage:
    python -m src.tasks.ball_detection.scripts.preview_augmentation
    python -m src.tasks.ball_detection.scripts.preview_augmentation preview.sample_indices=[0,1,2]
    python -m src.tasks.ball_detection.scripts.preview_augmentation preview.split=val

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/preview_augmentation.yaml`.
    - The script renders paired original and fully augmented contact sheets for selected clips.
    - Outputs are written under `outputs/ball_detection/augmentation_preview`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
    denormalize_tensor_images_imagenet,
)
from src.tasks.ball_detection.data.types import ClipWindow
from src.tasks.base.preview import resolve_sample_indices, resolve_split_file
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="preview_augmentation",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    output_dir = Path(str(cfg.preview.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = str(cfg.preview.split)
    split_file = resolve_split_file(cfg, split_name)
    datamodule = build_ball_detection_datamodule(cfg)
    base_dataset = datamodule.create_dataset(
        split_name=split_name,
        split_file=split_file,
        augmentation=None,
    )
    augmented_cfg = _build_augmented_config(cfg)
    augmented_datamodule = build_ball_detection_datamodule(augmented_cfg)
    augmented_dataset = augmented_datamodule.create_dataset(
        split_name=split_name,
        split_file=split_file,
        augmentation=BallDetectionAugmentation(
            OmegaConf.to_container(augmented_cfg.data.augmentation, resolve=True)
        ),
    )

    sample_indices = resolve_sample_indices(cfg, dataset_size=len(base_dataset), min_samples=1)
    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        torch.manual_seed(int(cfg.preview.seed) + sample_index)
        base_sample = base_dataset[sample_index]
        torch.manual_seed(int(cfg.preview.seed) + sample_index)
        augmented_sample = augmented_dataset[sample_index]
        window = base_dataset.windows[sample_index]
        sheet = _render_contact_sheet(
            base_sample=base_sample,
            augmented_sample=augmented_sample,
            window=window,
            cfg=cfg,
        )

        file_stem = _sample_stem(window=window, sample_index=sample_index)
        image_path = output_dir / f"{file_stem}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

        metadata = {
            "sample_index": sample_index,
            "clip_dir": str(window.clip_dir),
            "start_index": int(window.start_index),
            "frame_names": list(
                window.frame_names[
                    window.start_index : window.start_index + int(cfg.model.num_frames)
                ]
            ),
            "output_image": str(image_path),
            "split": split_name,
        }
        metadata_path = output_dir / f"{file_stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest.append(metadata)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(sample_indices)} augmentation preview(s) to {output_dir}")
    return 0


def _build_augmented_config(cfg: DictConfig) -> DictConfig:
    """Return a config copy with every configured augmentation enabled."""
    augmented_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if "data" not in augmented_cfg or "augmentation" not in augmented_cfg.data:
        raise ValueError("Expected data.augmentation to exist in the config.")

    augmentation_cfg = augmented_cfg.data.augmentation
    for name, transform_cfg in augmentation_cfg.items():
        if isinstance(transform_cfg, DictConfig) and "enabled" in transform_cfg:
            transform_cfg.enabled = True
        elif isinstance(transform_cfg, dict) and "enabled" in transform_cfg:
            transform_cfg["enabled"] = True
        elif name in {"camera_rotation", "horizontal_flip", "brightness_gain", "contrast", "gamma", "gaussian_noise", "gaussian_blur"}:
            raise ValueError(
                f"Expected data.augmentation.{name}.enabled to exist in the config."
            )
    return augmented_cfg


def _render_contact_sheet(
    *,
    base_sample: dict[str, torch.Tensor],
    augmented_sample: dict[str, torch.Tensor],
    window: ClipWindow,
    cfg: DictConfig,
) -> np.ndarray:
    """Render a two-row contact sheet for one sample window."""
    base_frames = _tensor_sequence_to_frames(base_sample["images"])
    augmented_frames = _tensor_sequence_to_frames(augmented_sample["images"])
    base_coords = base_sample["coords"].cpu().numpy()
    augmented_coords = augmented_sample["coords"].cpu().numpy()
    base_visibility = base_sample["visibility"].cpu().numpy()
    augmented_visibility = augmented_sample["visibility"].cpu().numpy()
    original_size_values = base_sample["original_size"].cpu().numpy().tolist()
    original_size = (int(original_size_values[0]), int(original_size_values[1]))

    draw_cfg = cfg.preview.draw
    annotated_base = _annotate_frames(
        frames=base_frames,
        coords=base_coords,
        visibility=base_visibility,
        original_size=original_size,
        radius=int(draw_cfg.radius),
        thickness=int(draw_cfg.thickness),
    )
    annotated_augmented = _annotate_frames(
        frames=augmented_frames,
        coords=augmented_coords,
        visibility=augmented_visibility,
        original_size=original_size,
        radius=int(draw_cfg.radius),
        thickness=int(draw_cfg.thickness),
    )

    frame_names = list(
        window.frame_names[window.start_index : window.start_index + len(annotated_base)]
    )
    tile_gap = int(cfg.preview.layout.tile_gap)
    header_height = int(cfg.preview.layout.header_height)
    row_gap = int(cfg.preview.layout.row_gap)
    text_scale = float(cfg.preview.layout.text_scale)
    text_thickness = int(cfg.preview.layout.text_thickness)
    background_values = [int(v) for v in cfg.preview.layout.background_rgb]
    if len(background_values) != 3:
        raise ValueError("preview.layout.background_rgb must contain exactly 3 values.")
    background_rgb = (
        background_values[0],
        background_values[1],
        background_values[2],
    )

    top_row = _compose_row(
        title="Original",
        frames=annotated_base,
        frame_names=frame_names,
        header_height=header_height,
        tile_gap=tile_gap,
        text_scale=text_scale,
        text_thickness=text_thickness,
        background_rgb=background_rgb,
    )
    bottom_row = _compose_row(
        title="All Augmentations Enabled",
        frames=annotated_augmented,
        frame_names=frame_names,
        header_height=header_height,
        tile_gap=tile_gap,
        text_scale=text_scale,
        text_thickness=text_thickness,
        background_rgb=background_rgb,
    )

    canvas_height = top_row.shape[0] + row_gap + bottom_row.shape[0]
    canvas_width = max(top_row.shape[1], bottom_row.shape[1])
    canvas: np.ndarray = np.full(
        (canvas_height, canvas_width, 3),
        background_rgb,
        dtype=np.uint8,
    )
    canvas[: top_row.shape[0], : top_row.shape[1]] = top_row
    canvas[top_row.shape[0] + row_gap :, : bottom_row.shape[1]] = bottom_row
    return canvas


def _tensor_sequence_to_frames(images: torch.Tensor) -> list[np.ndarray]:
    """Convert ``(T, 3, H, W)`` tensor images into RGB ``uint8`` frames."""
    images_to_render = images
    if _uses_imagenet_normalization(images):
        images_to_render = denormalize_tensor_images_imagenet(images)
    frames: list[np.ndarray] = []
    for image in images_to_render.cpu().numpy():
        frame = np.transpose(image, (1, 2, 0))
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        frames.append(frame)
    return frames


def _uses_imagenet_normalization(images: torch.Tensor) -> bool:
    """Heuristically detect normalized RGB tensors for preview rendering."""
    if images.ndim != 4 or images.shape[1] != 3:
        return False
    min_value = float(images.min().item())
    max_value = float(images.max().item())
    return min_value < 0.0 or max_value > 1.0


def _annotate_frames(
    *,
    frames: Sequence[np.ndarray],
    coords: np.ndarray,
    visibility: np.ndarray,
    original_size: tuple[int, int],
    radius: int,
    thickness: int,
) -> list[np.ndarray]:
    """Overlay ball positions on frames using the sample metadata."""
    original_width, original_height = original_size
    annotated: list[np.ndarray] = []
    for frame, frame_coords, frame_visibility in zip(
        frames,
        coords,
        visibility,
        strict=True,
    ):
        overlay = frame.copy()
        frame_height, frame_width = overlay.shape[:2]
        for coord, visible in zip(frame_coords, frame_visibility, strict=True):
            if float(visible) <= 0.0:
                continue
            x_pos = int(
                round(float(coord[0]) * frame_width / max(original_width, 1))
            )
            y_pos = int(
                round(float(coord[1]) * frame_height / max(original_height, 1))
            )
            cv2.circle(
                overlay,
                center=(x_pos, y_pos),
                radius=radius,
                color=(255, 80, 80),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        annotated.append(overlay)
    return annotated


def _compose_row(
    *,
    title: str,
    frames: Sequence[np.ndarray],
    frame_names: Sequence[str],
    header_height: int,
    tile_gap: int,
    text_scale: float,
    text_thickness: int,
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Compose one titled row of frames."""
    if not frames:
        raise ValueError("Expected at least one frame to compose a row.")

    frame_height, frame_width = frames[0].shape[:2]
    row_height = header_height + frame_height
    row_width = len(frames) * frame_width + max(len(frames) - 1, 0) * tile_gap
    row: np.ndarray = np.full(
        (row_height, row_width, 3),
        background_rgb,
        dtype=np.uint8,
    )

    _put_label(
        row,
        text=title,
        origin=(10, max(18, header_height - 10)),
        scale=text_scale,
        thickness=text_thickness,
        color=(235, 235, 235),
    )
    for frame_index, (frame, frame_name) in enumerate(zip(frames, frame_names, strict=True)):
        x0 = frame_index * (frame_width + tile_gap)
        row[header_height : header_height + frame_height, x0 : x0 + frame_width] = frame
        _put_label(
            row,
            text=f"{frame_index:02d} {frame_name}",
            origin=(x0 + 8, max(18, header_height - 10)),
            scale=max(text_scale - 0.08, 0.3),
            thickness=text_thickness,
            color=(190, 190, 190),
        )
    return row


def _put_label(
    image: np.ndarray,
    *,
    text: str,
    origin: tuple[int, int],
    scale: float,
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a single text label with anti-aliased rendering."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _sample_stem(*, window: ClipWindow, sample_index: int) -> str:
    """Build a stable output file stem for one window."""
    clip_name = window.clip_dir.name
    game_name = window.clip_dir.parent.name
    return f"{sample_index:06d}_{game_name}_{clip_name}_start{window.start_index:04d}"


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
