"""Evaluate a court detection model on a random subset of the dataset.

Loads a Lightning checkpoint, samples N items from the selected split, runs
inference, and saves task-specific visualisations under the configured output
directory.

Usage:
    python -m src.tasks.court_detection.scripts.eval task=kp checkpoint=<path/to/model.ckpt>
    python -m src.tasks.court_detection.scripts.eval task=seg checkpoint=<ckpt> num_samples=20 split=val
    python -m src.tasks.court_detection.scripts.eval task=line checkpoint=<ckpt> device=cpu

Notes:
    - Hydra loads configuration from ``src/tasks/court_detection/configs/eval.yaml``.
    - ``task`` must be specified explicitly and must match the checkpoint task.
    - ``num_samples=-1`` evaluates the entire split.
    - Seg/line overlays share the same alpha-blend helper.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset
from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset
from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset
from src.tasks.court_detection.inference import CourtDetectionPredictor

_SEG_PALETTE = np.array([
    [0, 0, 0],
    [255, 100, 100],
    [100, 100, 255],
    [100, 255, 100],
    [255, 255, 100],
    [255, 100, 255],
    [100, 255, 255],
], dtype=np.uint8)


def _save_heatmap_grid(
    heatmaps: torch.Tensor,
    save_path: Path,
    ncols: int = 7,
) -> None:
    """Save all keypoint heatmaps as a single grid PNG."""
    k, h, w = heatmaps.shape
    nrows = math.ceil(k / ncols)
    grid = Image.new("L", (ncols * w, nrows * h), color=0)

    for idx in range(k):
        ch = heatmaps[idx].float()
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch = torch.zeros_like(ch)
        arr = (ch.numpy() * 255).astype(np.uint8)
        tile = Image.fromarray(arr, mode="L")
        row, col = divmod(idx, ncols)
        grid.paste(tile, (col * w, row * h))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(save_path)


def _save_kp_overlay(
    original_image: Image.Image,
    keypoints: torch.Tensor,
    save_path: Path,
    radius: int = 6,
) -> None:
    """Save the original image with predicted keypoints overlaid."""
    img = original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for x, y in keypoints.numpy():
        x0, y0 = float(x) - radius, float(y) - radius
        x1, y1 = float(x) + radius, float(y) + radius
        draw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def _colorize_seg_mask(mask: np.ndarray) -> np.ndarray:
    """Map label mask ``[H, W]`` to an RGB image."""
    return _SEG_PALETTE[mask]


def _colorize_line_heatmap(mask: np.ndarray) -> np.ndarray:
    """Map line probability heatmap ``[H, W]`` to an RGB image."""
    heatmap_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(
        cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )


def _save_rgb_image(image: np.ndarray, save_path: Path) -> None:
    """Save an RGB uint8 image."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(save_path)


def _alpha_blend_overlay(
    original_image: Image.Image,
    overlay_rgb: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Alpha-blend an RGB overlay onto the original image."""
    base = np.asarray(original_image.convert("RGB"), dtype=np.uint8)
    if overlay_rgb.shape[:2] != base.shape[:2]:
        overlay_rgb = cv2.resize(
            overlay_rgb,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    blended = (
        base.astype(np.float32) * (1.0 - alpha) +
        overlay_rgb.astype(np.float32) * alpha
    )
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _build_dataset(cfg: DictConfig) -> CourtKPDataset | CourtSegDataset | CourtLineDataset:
    """Build the dataset matching the configured task."""
    aug_cfg = dict(cfg.get("augmentation", {}))
    if cfg.task == "kp":
        return CourtKPDataset(
            root=cfg.data_dir,
            split=cfg.split,
            is_train=False,
            config=aug_cfg,
        )
    if cfg.task == "seg":
        return CourtSegDataset(
            root=cfg.data_dir,
            split=cfg.split,
            is_train=False,
            config=aug_cfg,
        )
    if cfg.task == "line":
        return CourtLineDataset(
            root=cfg.data_dir,
            split=cfg.split,
            is_train=False,
            config=aug_cfg,
            mask_dir_name=str(cfg.get("mask_dir_name", "line_masks")),
        )
    raise ValueError(f"Unsupported task: {cfg.task}")


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="eval",
)
def main(cfg: DictConfig) -> None:
    """Evaluate court detection and save task-specific visual results."""
    if not cfg.checkpoint:
        raise ValueError(
            "checkpoint must be specified. "
            "Pass checkpoint=<path/to/model.ckpt> on the command line."
        )

    if not cfg.task:
        raise ValueError("task must be specified explicitly: kp, seg, or line.")

    output_dir = Path(cfg.output_dir)
    predictor = CourtDetectionPredictor.load_from_checkpoint(
        cfg.checkpoint,
        device=cfg.device,
    )
    if predictor.task != cfg.task:
        raise ValueError(
            f"Configured task '{cfg.task}' does not match checkpoint task '{predictor.task}'."
        )

    dataset = _build_dataset(cfg)
    n = len(dataset)
    num_samples = int(cfg.num_samples)
    if num_samples < 0 or num_samples >= n:
        indices = list(range(n))
    else:
        indices = random.sample(range(n), num_samples)

    images_dir = Path(cfg.data_dir) / "images"

    for idx in indices:
        entry = dataset._entries[idx]
        image_id: str = entry["id"]

        img_path = images_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = images_dir / f"{image_id}.jpg"
        original_image = Image.open(img_path).convert("RGB")

        result = predictor.predict(original_image, return_logits=True)

        if cfg.task == "kp":
            _save_heatmap_grid(
                result["logits"],
                output_dir / "heatmaps" / f"{image_id}_heatmap_grid.png",
            )
            _save_kp_overlay(
                original_image,
                result["keypoints"],
                output_dir / "overlays" / f"{image_id}_overlay.png",
            )
            continue

        if cfg.task == "seg":
            heatmap_rgb = _colorize_seg_mask(result["mask"].numpy().astype(np.uint8))
        else:
            heatmap_rgb = _colorize_line_heatmap(result["mask"].numpy())

        _save_rgb_image(
            heatmap_rgb,
            output_dir / "heatmaps" / f"{image_id}_heatmap.png",
        )
        overlay = _alpha_blend_overlay(original_image, heatmap_rgb, alpha=float(cfg.alpha))
        overlay_path = output_dir / "overlays" / f"{image_id}_overlay.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)

    print(f"Saved {len(indices)} results to {output_dir}")


if __name__ == "__main__":
    main()
