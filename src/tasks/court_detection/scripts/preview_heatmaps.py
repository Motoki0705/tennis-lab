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

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from src.utils.data.heatmaps import generate_gaussian_heatmaps, heatmaps_to_argmax

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


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
    entries = json.loads((data_dir / f"data_{cfg.preview.split}.json").read_text(encoding="utf-8"))
    sample_indices = _resolve_sample_indices(cfg, len(entries))

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

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} court heatmap preview(s) to {output_dir}")
    return 0


def _resolve_sample_indices(cfg: DictConfig, dataset_size: int) -> list[int]:
    explicit = [int(value) for value in cfg.preview.sample_indices]
    if explicit:
        sample_indices = explicit
    else:
        sample_indices = list(range(min(int(cfg.preview.max_samples), dataset_size)))
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= dataset_size:
            raise IndexError(f"preview sample_index={sample_index} is out of range for dataset size {dataset_size}.")
    return sample_indices


def _load_image(data_dir: Path, image_id: str) -> np.ndarray:
    for extension in (".png", ".jpg", ".jpeg"):
        image_path = data_dir / "images" / f"{image_id}{extension}"
        if image_path.exists():
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"))
    raise FileNotFoundError(f"Image not found for image_id={image_id!r} under {data_dir / 'images'}")


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


def _draw_point(
    image: np.ndarray,
    center_xy: tuple[float, float],
    *,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x_px = int(round(center_xy[0] * max(width - 1, 0)))
    y_px = int(round(center_xy[1] * max(height - 1, 0)))
    cv2.circle(image, (x_px, y_px), radius, color, thickness=thickness, lineType=cv2.LINE_AA)


def _compose_row(
    panels: list[np.ndarray],
    titles: list[str],
    cfg: DictConfig,
) -> np.ndarray:
    tile_gap = int(cfg.preview.layout.tile_gap)
    header_height = int(cfg.preview.layout.header_height)
    text_scale = float(cfg.preview.layout.text_scale)
    text_thickness = int(cfg.preview.layout.text_thickness)
    background_rgb = tuple(int(v) for v in cfg.preview.layout.background_rgb)

    height, width = panels[0].shape[:2]
    row_width = len(panels) * width + (len(panels) - 1) * tile_gap
    canvas = np.full((header_height + height, row_width, 3), background_rgb, dtype=np.uint8)

    cursor_x = 0
    for panel, title in zip(panels, titles, strict=True):
        canvas[header_height:, cursor_x : cursor_x + width] = panel
        cv2.putText(
            canvas,
            title,
            (cursor_x + 6, max(header_height - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (245, 245, 245),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        cursor_x += width + tile_gap
    return canvas


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
