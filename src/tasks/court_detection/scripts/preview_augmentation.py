"""Render original-vs-augmented previews for court-detection training samples.

Usage:
    python -m src.tasks.court_detection.scripts.preview_augmentation
    python -m src.tasks.court_detection.scripts.preview_augmentation data/processing=all
    python -m src.tasks.court_detection.scripts.preview_augmentation data/source=synthetic_court preview.split=val
    python -m src.tasks.court_detection.scripts.preview_augmentation preview.sample_indices=[0,8,16]

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/preview_augmentation.yaml`.
    - Court augmentation has no per-transform `enabled` flags; the "augmented"
      panels are produced by building the dataset with `is_train=True` (full
      training pipeline) while the "original" panel uses `is_train=False`
      (deterministic validation resize only).
    - Each sample renders one original panel plus `preview.num_augmented`
      independently seeded augmentation draws, with task-specific annotations
      (kp: keypoint circles, seg/line: mask overlay).
    - Outputs are resolved beneath `paths.output_root`.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping, Sized
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.base.visualization.preview import (
    compose_titled_row,
    resolve_sample_indices,
)
from src.tasks.court_detection.configuration import (
    CourtDataConfig,
    validate_paths_boundary,
)
from src.tasks.court_detection.data.contracts import CourtSourceSplit
from src.tasks.court_detection.data.dataset import CourtDetectionDataset
from src.tasks.court_detection.data.processing.factory import (
    build_court_processing_pipeline,
)
from src.tasks.court_detection.visualization.rendering.common import (
    colorize_seg_mask,
    denormalize_tensor_to_rgb,
)
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json

_LINE_OVERLAY_RGB = (255, 96, 96)
_KP_COLOR_RGB = (255, 80, 80)
_BOUNDARY = "court_detection.preview_augmentation"


def _runtime(cfg: DictConfig) -> tuple[Path, CourtDataConfig]:
    root, resolver = validate_paths_boundary(cfg, expected_sections={"data", "preview"})
    data = CourtDataConfig.from_mapping(
        require_config_mapping(root, "data", path="configuration"), resolver=resolver
    )
    preview = require_config_mapping(root, "preview", path="configuration")
    expected = {
        "split",
        "sample_indices",
        "max_samples",
        "num_augmented",
        "seed",
        "output_dir",
        "draw",
        "layout",
    }
    if set(preview) != expected:
        raise ValueError(f"preview requires exactly {sorted(expected)}.")
    for key in ("max_samples", "num_augmented", "seed"):
        require_config_value(preview, key, int, path="preview")
    if cast("int", preview["max_samples"]) <= 0:
        raise ValueError("preview.max_samples must be positive.")
    if cast("int", preview["num_augmented"]) < 1:
        raise ValueError("preview.num_augmented must be >= 1.")
    split = cast("str", require_config_value(preview, "split", str, path="preview"))
    if split not in {"train", "val"}:
        raise ValueError("preview.split must be train or val.")
    sample_indices = cast(
        "list[object] | tuple[object, ...]",
        require_config_value(preview, "sample_indices", (list, tuple), path="preview"),
    )
    if any(type(index) is not int or index < 0 for index in sample_indices):
        raise ValueError("preview.sample_indices must contain non-negative integers.")
    draw = require_config_mapping(preview, "draw", path="preview")
    layout = require_config_mapping(preview, "layout", path="preview")
    if set(draw) != {"kp_radius", "kp_thickness", "mask_alpha"}:
        raise ValueError("preview.draw has an invalid field set.")
    if set(layout) != {
        "tile_gap",
        "header_height",
        "text_scale",
        "text_thickness",
        "background_rgb",
    }:
        raise ValueError("preview.layout has an invalid field set.")
    for key in ("kp_radius", "kp_thickness"):
        require_config_value(draw, key, int, path="preview.draw")
    require_config_value(draw, "mask_alpha", (float, int), path="preview.draw")
    for key in ("tile_gap", "header_height", "text_thickness"):
        require_config_value(layout, key, int, path="preview.layout")
    require_config_value(layout, "text_scale", (float, int), path="preview.layout")
    require_config_value(layout, "background_rgb", list, path="preview.layout")
    background = cast("list[object]", layout["background_rgb"])
    if len(background) != 3 or any(
        type(channel) is not int or not 0 <= channel <= 255 for channel in background
    ):
        raise ValueError("preview.layout.background_rgb must be three RGB integers.")
    if any(cast("int", draw[key]) <= 0 for key in ("kp_radius", "kp_thickness")):
        raise ValueError("preview keypoint draw sizes must be positive.")
    mask_alpha = float(cast("float | int", draw["mask_alpha"]))
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("preview.draw.mask_alpha must be in [0, 1].")
    if cast("int", layout["tile_gap"]) < 0 or any(
        cast("int", layout[key]) <= 0 for key in ("header_height", "text_thickness")
    ):
        raise ValueError("preview.layout sizes are invalid.")
    if float(cast("float | int", layout["text_scale"])) <= 0:
        raise ValueError("preview.layout.text_scale must be positive.")
    output_dir = cast(
        "str", require_config_value(preview, "output_dir", str, path="preview")
    )
    if not output_dir:
        raise ValueError("preview.output_dir must not be empty.")
    return resolver.resolve(
        PathRole.OUTPUT,
        output_dir,
    ), data


def _validate_boundary(cfg: DictConfig) -> None:
    _runtime(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="preview_augmentation",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    output_dir, data = _runtime(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = str(cfg.preview.split)
    base_dataset = _dataset(data, split=split_name, is_train=False)
    augmented_dataset = _dataset(data, split=split_name, is_train=True)

    target_kinds = tuple(target.kind for target in data.processing.targets)
    seed = int(cfg.preview.seed)
    num_augmented = int(cfg.preview.num_augmented)
    if num_augmented < 1:
        raise ValueError("preview.num_augmented must be >= 1.")

    sample_indices = resolve_sample_indices(
        cfg, dataset_size=len(cast("Sized", base_dataset)), min_samples=1
    )
    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        _seed_all(seed + sample_index)
        base_sample = base_dataset[sample_index]
        panels = [_annotate_sample(base_sample, target_kinds=target_kinds, cfg=cfg)]
        titles = ["original"]
        for variant in range(num_augmented):
            _seed_all(seed + sample_index * 1009 + variant + 1)
            augmented_sample = augmented_dataset[sample_index]
            panels.append(
                _annotate_sample(
                    augmented_sample,
                    target_kinds=target_kinds,
                    cfg=cfg,
                )
            )
            titles.append(f"augmented #{variant}")

        panels = _pad_panels_to_common_size(panels, cfg)
        sheet = compose_titled_row(panels, titles, cfg)

        sample_id = str(base_sample["sample_id"])
        file_stem = f"{sample_index:06d}_{sample_id.replace(':', '_')}"
        image_path = output_dir / f"{file_stem}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

        metadata = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "targets": list(target_kinds),
            "split": split_name,
            "num_augmented": num_augmented,
            "output_image": str(image_path),
        }
        save_json(metadata, output_dir / f"{file_stem}.json")
        manifest.append(metadata)

    save_json(manifest, output_dir / "manifest.json")
    print(f"Saved {len(manifest)} augmentation preview(s) to {output_dir}")
    return 0


def _seed_all(seed: int) -> None:
    """Seed every RNG used by the court augmentation pipeline."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _dataset(
    data: CourtDataConfig,
    *,
    split: str,
    is_train: bool,
) -> CourtDetectionDataset:
    if split not in {"train", "val"}:
        raise ValueError("Preview split must be train or val.")
    pipeline = build_court_processing_pipeline(data, is_train=is_train)
    records = pipeline.input_layer.records(cast("CourtSourceSplit", split))
    return CourtDetectionDataset(records, pipeline=pipeline)


def _annotate_sample(
    sample: dict[str, Any],
    *,
    target_kinds: tuple[str, ...],
    cfg: DictConfig,
) -> np.ndarray:
    """Render every selected target over one shared RGB geometry."""
    image = cast("torch.Tensor", sample["image"])
    rgb = denormalize_tensor_to_rgb(image)
    targets = cast("Mapping[str, object]", sample["targets"])
    for kind in target_kinds:
        if kind == "kp":
            rgb = _overlay_keypoints(rgb, targets[kind], cfg)
        elif kind == "seg":
            rgb = _overlay_seg_mask(rgb, targets[kind], cfg)
        elif kind == "line":
            rgb = _overlay_line_mask(rgb, targets[kind], cfg)
        else:  # pragma: no cover - strict configuration rejects this
            raise ValueError(f"Unknown Court target: {kind!r}")
    return cast("np.ndarray", rgb)


def _overlay_keypoints(rgb: np.ndarray, value: object, cfg: DictConfig) -> np.ndarray:
    """Draw pixel-space court keypoints onto the panel."""
    overlay = rgb.copy()
    payload = cast("Mapping[str, torch.Tensor]", value)
    keypoints = payload["points_xy"].cpu().numpy()
    visible = payload["point_visible"].cpu().numpy()
    height, width = overlay.shape[:2]
    for x_pos, y_pos in keypoints[visible]:
        if not (0.0 <= float(x_pos) < width and 0.0 <= float(y_pos) < height):
            continue
        cv2.circle(
            overlay,
            (int(round(float(x_pos))), int(round(float(y_pos)))),
            int(cfg.preview.draw.kp_radius),
            _KP_COLOR_RGB,
            thickness=int(cfg.preview.draw.kp_thickness),
            lineType=cv2.LINE_AA,
        )
    return overlay


def _overlay_seg_mask(rgb: np.ndarray, value: object, cfg: DictConfig) -> np.ndarray:
    """Blend the colorized segmentation mask over foreground pixels."""
    mask = cast("torch.Tensor", value).cpu().numpy()
    colored = colorize_seg_mask(mask)
    return _blend_where(rgb, colored, mask > 0, float(cfg.preview.draw.mask_alpha))


def _overlay_line_mask(rgb: np.ndarray, value: object, cfg: DictConfig) -> np.ndarray:
    """Tint white-line pixels with a solid overlay color."""
    mask = cast("torch.Tensor", value).cpu().numpy()[0]
    colored = np.zeros_like(rgb)
    colored[:, :] = _LINE_OVERLAY_RGB
    return _blend_where(rgb, colored, mask > 0.5, float(cfg.preview.draw.mask_alpha))


def _blend_where(
    rgb: np.ndarray, colored: np.ndarray, region: np.ndarray, alpha: float
) -> np.ndarray:
    """Alpha-blend ``colored`` into ``rgb`` only where ``region`` is true."""
    overlay = rgb.copy()
    blended = (
        rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    )
    overlay[region] = np.clip(blended[region], 0.0, 255.0).astype(np.uint8)
    return overlay


def _pad_panels_to_common_size(
    panels: list[np.ndarray], cfg: DictConfig
) -> list[np.ndarray]:
    """Pad panels to a shared canvas so they can be composed in one row.

    Augmented crops/resizes produce per-panel sizes, while the shared row
    composer requires equal panels; padding (instead of resizing) keeps the
    augmented geometry visible at true scale.
    """
    background_rgb = tuple(int(v) for v in cfg.preview.layout.background_rgb)
    max_height = max(panel.shape[0] for panel in panels)
    max_width = max(panel.shape[1] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        canvas = np.full((max_height, max_width, 3), background_rgb, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(canvas)
    return padded


if __name__ == "__main__":
    sys.exit(main())
