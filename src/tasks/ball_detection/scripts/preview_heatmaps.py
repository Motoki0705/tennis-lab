"""Render qualitative ball heatmap previews for multiple sigma ratios.

Usage:
    python -m src.tasks.ball_detection.scripts.preview_heatmaps
    python -m src.tasks.ball_detection.scripts.preview_heatmaps preview.ratios=[0.004,0.008,0.016]
    python -m src.tasks.ball_detection.scripts.preview_heatmaps preview.sample_indices=[10,20] preview.split=val

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/preview_heatmaps.yaml`.
    - The script renders one representative frame per selected sample window.
    - Each panel overlays a generated heatmap and argmax point on the source frame.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.data import build_ball_detection_datamodule
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

    split_name = str(cfg.preview.split)
    datamodule = build_ball_detection_datamodule(cfg)
    dataset = datamodule.create_dataset(
        split_name=split_name,
        split_file=_resolve_split_file(cfg, split_name),
        augmentation=None,
    )
    sample_indices = _resolve_sample_indices(cfg, len(dataset))

    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        frame_index = _select_frame_index(sample["visibility"])
        image = _tensor_to_image(sample["images"][frame_index])
        height, width = image.shape[:2]

        original_size = sample["original_size"].cpu().numpy()
        original_w = float(original_size[0])
        original_h = float(original_size[1])
        centers_xy = _frame_centers_xy(
            sample["coords"][frame_index],
            original_w=original_w,
            original_h=original_h,
        )
        visibility = sample["visibility"][frame_index].detach().cpu().tolist()
        visible = any(float(value) > 0.5 for value in visibility)

        panels = [_annotate_original(image, centers_xy, visibility, cfg)]
        ratios = [float(value) for value in cfg.preview.ratios]
        ratio_records: list[dict[str, Any]] = []
        for sigma_ratio in ratios:
            instance_heatmaps = generate_gaussian_heatmaps(
                size_hw=(height, width),
                centers_xy=centers_xy,
                sigma_ratio=sigma_ratio,
                visibility=visibility,
            )
            heatmap = (
                instance_heatmaps.amax(dim=0)
                if instance_heatmaps.numel() > 0
                else torch.zeros((height, width), dtype=torch.float32)
            )
            argmax_xy, peak_value = heatmaps_to_argmax(heatmap)
            panel = _render_ratio_panel(
                image=image,
                heatmap=heatmap,
                sigma_ratio=sigma_ratio,
                gt_centers_xy=centers_xy,
                gt_visibility=visibility,
                argmax_xy=tuple(float(v) for v in argmax_xy.tolist()),
                peak_value=float(peak_value.item()),
                cfg=cfg,
            )
            panels.append(panel)
            ratio_records.append(
                {
                    "sigma_ratio": sigma_ratio,
                    "argmax_xy": tuple(float(v) for v in argmax_xy.tolist()),
                    "peak_value": float(peak_value.item()),
                }
            )

        canvas = _compose_row(panels, ["original", *[f"ratio={ratio:.4f}" for ratio in ratios]], cfg)
        file_stem = f"sample_{sample_index:05d}_frame_{frame_index:02d}"
        image_path = output_dir / f"{file_stem}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        window = dataset.windows[sample_index]
        manifest.append(
            {
                "sample_index": sample_index,
                "frame_index": frame_index,
                "clip_dir": str(window.clip_dir),
                "frame_name": window.frame_names[window.start_index + frame_index],
                "output_image": str(image_path),
                "visible": visible,
                "centers_xy": centers_xy,
                "ratios": ratio_records,
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} ball heatmap preview(s) to {output_dir}")
    return 0


def _resolve_split_file(cfg: DictConfig, split_name: str) -> str:
    split_cfg = cfg.data.split
    key = f"{split_name}_file"
    if key not in split_cfg:
        available = ", ".join(sorted(split_cfg.keys()))
        raise ValueError(f"Unknown preview.split={split_name!r}. Available: {available}")
    return str(split_cfg[key])


def _resolve_sample_indices(cfg: DictConfig, dataset_size: int) -> list[int]:
    explicit = [int(value) for value in cfg.preview.sample_indices]
    if explicit:
        sample_indices = explicit
    else:
        sample_indices = list(range(min(int(cfg.preview.max_samples), dataset_size)))
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= dataset_size:
            raise IndexError(
                f"preview sample_index={sample_index} is out of range for "
                f"dataset size {dataset_size}."
            )
    return sample_indices


def _select_frame_index(visibility: torch.Tensor) -> int:
    frame_visibility = (
        (visibility > 0.5).any(dim=-1)
        if visibility.ndim == 2
        else visibility > 0.5
    )
    visible_indices = torch.nonzero(frame_visibility, as_tuple=False).flatten()
    if len(visible_indices) > 0:
        return int(visible_indices[0].item())
    return 0


def _frame_centers_xy(
    coords: torch.Tensor,
    *,
    original_w: float,
    original_h: float,
) -> list[tuple[float, float]]:
    return [
        (
            float(coord[0].item() / max(original_w - 1.0, 1.0)),
            float(coord[1].item() / max(original_h - 1.0, 1.0)),
        )
        for coord in coords
    ]


def _tensor_to_image(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return image_np


def _annotate_original(
    image: np.ndarray,
    centers_xy: list[tuple[float, float]],
    visibility: list[float],
    cfg: DictConfig,
) -> np.ndarray:
    canvas = image.copy()
    for center_xy, visible in zip(centers_xy, visibility, strict=True):
        if float(visible) <= 0.5:
            continue
        _draw_point(
            canvas,
            center_xy,
            radius=int(cfg.preview.draw.gt_radius),
            color=(255, 80, 80),
            thickness=int(cfg.preview.draw.thickness),
        )
    return canvas


def _render_ratio_panel(
    *,
    image: np.ndarray,
    heatmap: torch.Tensor,
    sigma_ratio: float,
    gt_centers_xy: list[tuple[float, float]],
    gt_visibility: list[float],
    argmax_xy: tuple[float, float],
    peak_value: float,
    cfg: DictConfig,
) -> np.ndarray:
    overlay = image.copy()
    heatmap_np = np.clip(heatmap.detach().cpu().numpy(), 0.0, 1.0)
    heatmap_cm = cv2.applyColorMap((heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_cm = cv2.cvtColor(heatmap_cm, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.55, heatmap_cm, 0.45, 0.0)

    if peak_value > 0:
        _draw_point(
            overlay,
            argmax_xy,
            radius=int(cfg.preview.draw.argmax_radius),
            color=(80, 255, 120),
            thickness=int(cfg.preview.draw.thickness),
        )
    for center_xy, visible in zip(gt_centers_xy, gt_visibility, strict=True):
        if float(visible) <= 0.5:
            continue
        _draw_point(
            overlay,
            center_xy,
            radius=int(cfg.preview.draw.gt_radius),
            color=(255, 80, 80),
            thickness=1,
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
