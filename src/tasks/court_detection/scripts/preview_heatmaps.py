"""Render qualitative court-keypoint heatmap previews for multiple sigma ratios.

Usage:
    python -m src.tasks.court_detection.scripts.preview_heatmaps
    python -m src.tasks.court_detection.scripts.preview_heatmaps preview.ratios=[0.004,0.008,0.016]
    python -m src.tasks.court_detection.scripts.preview_heatmaps preview.sample_indices=[10,20] preview.split=val

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/preview_heatmaps.yaml`.
    - The script reads raw court keypoint annotations and images without training augmentation.
    - Each panel overlays the max-pooled heatmap plus decoded argmax points for all keypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from src.tasks.base.preview import (
    compose_titled_row as _compose_row,
)
from src.tasks.base.preview import (
    draw_normalized_point as _draw_point,
)
from src.tasks.base.preview import (
    resolve_sample_indices,
)
from src.utils.data.heatmaps import generate_gaussian_heatmaps, heatmaps_to_argmax
from src.utils.hydra import hydra_main
from src.utils.io import find_existing_file, load_json, save_json


@hydra_main(
    config_path="../configs",
    config_name="preview_heatmaps",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    output_dir = Path(str(cfg.preview.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(str(cfg.data.data_dir)).expanduser()
    entries = load_json(data_dir / f"data_{cfg.preview.split}.json")
    sample_indices = resolve_sample_indices(cfg, len(entries))

    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        entry = entries[sample_index]
        image_id = str(entry["id"])
        image = _load_image(data_dir, image_id)
        height, width = image.shape[:2]
        keypoints = np.asarray(entry["kps"], dtype=np.float32)
        centers_xy = np.stack(
            [
                keypoints[:, 0] / max(width - 1, 1),
                keypoints[:, 1] / max(height - 1, 1),
            ],
            axis=-1,
        )

        panels = [_annotate_original(image, centers_xy, cfg)]
        ratios = [float(value) for value in cfg.preview.ratios]
        ratio_records: list[dict[str, Any]] = []
        for sigma_ratio in ratios:
            heatmaps = generate_gaussian_heatmaps(
                size_hw=(height, width),
                centers_xy=centers_xy,
                sigma_ratio=sigma_ratio,
            )
            argmax_xy, peak_values = heatmaps_to_argmax(heatmaps)
            panel = _render_ratio_panel(
                image=image,
                heatmaps=heatmaps,
                gt_centers_xy=centers_xy,
                argmax_xy=argmax_xy.detach().cpu().numpy(),
                peak_values=peak_values.detach().cpu().numpy(),
                cfg=cfg,
            )
            panels.append(panel)
            ratio_records.append(
                {
                    "sigma_ratio": sigma_ratio,
                    "mean_peak_value": float(peak_values.mean().item()),
                }
            )

        canvas = _compose_row(panels, ["original", *[f"ratio={ratio:.4f}" for ratio in ratios]], cfg)
        image_path = output_dir / f"sample_{sample_index:05d}_{image_id}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        manifest.append(
            {
                "sample_index": sample_index,
                "image_id": image_id,
                "output_image": str(image_path),
                "ratios": ratio_records,
            }
        )

    save_json(manifest, output_dir / "manifest.json")
    print(f"Saved {len(manifest)} court heatmap preview(s) to {output_dir}")
    return 0


def _load_image(data_dir: Path, image_id: str) -> np.ndarray:
    image_path = find_existing_file(data_dir / "images", image_id, (".png", ".jpg", ".jpeg"))
    if image_path is None:
        raise FileNotFoundError(f"Image not found for image_id={image_id!r} under {data_dir / 'images'}")
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def _annotate_original(image: np.ndarray, centers_xy: np.ndarray, cfg: DictConfig) -> np.ndarray:
    canvas = image.copy()
    for center_xy in centers_xy:
        _draw_point(
            canvas,
            tuple(float(v) for v in center_xy.tolist()),
            radius=int(cfg.preview.draw.gt_radius),
            color=(255, 80, 80),
            thickness=int(cfg.preview.draw.thickness),
        )
    return canvas


def _render_ratio_panel(
    *,
    image: np.ndarray,
    heatmaps: np.ndarray | Any,
    gt_centers_xy: np.ndarray,
    argmax_xy: np.ndarray,
    peak_values: np.ndarray,
    cfg: DictConfig,
) -> np.ndarray:
    overlay = image.copy()
    heatmaps_np = np.asarray(heatmaps.detach().cpu().numpy() if hasattr(heatmaps, "detach") else heatmaps)
    pooled = np.clip(heatmaps_np.max(axis=0), 0.0, 1.0)
    heatmap_cm = cv2.applyColorMap((pooled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_cm = cv2.cvtColor(heatmap_cm, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.55, heatmap_cm, 0.45, 0.0)

    for center_xy in gt_centers_xy:
        _draw_point(
            overlay,
            tuple(float(v) for v in center_xy.tolist()),
            radius=int(cfg.preview.draw.gt_radius),
            color=(255, 80, 80),
            thickness=1,
        )
    for center_xy, peak_value in zip(argmax_xy, peak_values, strict=True):
        if float(peak_value) <= 0:
            continue
        _draw_point(
            overlay,
            tuple(float(v) for v in center_xy.tolist()),
            radius=int(cfg.preview.draw.argmax_radius),
            color=(80, 255, 120),
            thickness=int(cfg.preview.draw.thickness),
        )
    return overlay


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
