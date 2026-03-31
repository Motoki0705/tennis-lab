"""Generate contact sheets for ball-detection sequence augmentations.

Usage:
    python -m src.tasks.ball_detection.scripts.visualize_augmentation
    python -m src.tasks.ball_detection.scripts.visualize_augmentation preview.sample_indices=[0,1,2]
    python -m src.tasks.ball_detection.scripts.visualize_augmentation preview.split=val

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/visualize_augmentation.yaml`.
    - The script renders paired original and fully augmented contact sheets for selected clips.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.data.argumentation import BallDetectionArgumentation
from src.tasks.ball_detection.data.dataset import BallDetectionDataset, ClipWindow

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(
    config_path="../configs",
    config_name="visualize_augmentation",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    output_dir = Path(str(cfg.preview.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = str(cfg.preview.split)
    split_file = _resolve_split_file(cfg, split_name)
    base_dataset = BallDetectionDataset(
        data_dir=cfg.data.data_dir,
        split_file=split_file,
        config=cfg,
        argumentation=None,
    )
    augmented_cfg = _build_augmented_config(cfg)
    augmented_dataset = BallDetectionDataset(
        data_dir=cfg.data.data_dir,
        split_file=split_file,
        config=augmented_cfg,
        argumentation=BallDetectionArgumentation(
            OmegaConf.to_container(augmented_cfg.data.augmentation, resolve=True)
        ),
    )

    sample_indices = _resolve_sample_indices(cfg, dataset_size=len(base_dataset))
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


def _resolve_split_file(cfg: DictConfig, split_name: str) -> str:
    """Return the split file path from ``cfg.data.split``."""
    split_cfg = cfg.data.split
    key = f"{split_name}_file"
    if key not in split_cfg:
        available = ", ".join(sorted(split_cfg.keys()))
        raise ValueError(f"Unknown preview.split={split_name!r}. Available: {available}")
    return str(split_cfg[key])


def _resolve_sample_indices(cfg: DictConfig, *, dataset_size: int) -> list[int]:
    """Return validated sample indices to render."""
    explicit = [int(value) for value in cfg.preview.sample_indices]
    if explicit:
        sample_indices = explicit
    else:
        max_samples = max(int(cfg.preview.max_samples), 1)
        sample_indices = list(range(min(max_samples, dataset_size)))

    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= dataset_size:
            raise IndexError(
                f"preview sample_index={sample_index} is out of range for dataset size {dataset_size}."
            )
    return sample_indices


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
    original_size = tuple(int(v) for v in base_sample["original_size"].cpu().numpy().tolist())

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
    background_rgb = tuple(int(v) for v in cfg.preview.layout.background_rgb)

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
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        background_rgb,
        dtype=np.uint8,
    )
    canvas[: top_row.shape[0], : top_row.shape[1]] = top_row
    canvas[top_row.shape[0] + row_gap :, : bottom_row.shape[1]] = bottom_row
    return canvas


def _tensor_sequence_to_frames(images: torch.Tensor) -> list[np.ndarray]:
    """Convert ``(T, 3, H, W)`` tensor images into RGB ``uint8`` frames."""
    frames: list[np.ndarray] = []
    for image in images.cpu().numpy():
        frame = np.transpose(image, (1, 2, 0))
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        frames.append(frame)
    return frames


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
    for frame, coord, vis in zip(frames, coords, visibility, strict=True):
        overlay = frame.copy()
        if float(vis) > 0.0:
            frame_height, frame_width = overlay.shape[:2]
            x_pos = int(round(float(coord[0]) * frame_width / max(original_width, 1)))
            y_pos = int(round(float(coord[1]) * frame_height / max(original_height, 1)))
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
    row = np.full((row_height, row_width, 3), background_rgb, dtype=np.uint8)

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
