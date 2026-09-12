"""Render original-vs-augmented previews for court-detection training samples.

Usage:
    python -m src.tasks.court_detection.scripts.preview_augmentation
    python -m src.tasks.court_detection.scripts.preview_augmentation data/processing=all
    python -m src.tasks.court_detection.scripts.preview_augmentation data/source=synthetic_court preview.split=val
    python -m src.tasks.court_detection.scripts.preview_augmentation data/source=synthetic_court data/augmentation=pose_safe preview.require_pose=true
    python -m src.tasks.court_detection.scripts.preview_augmentation preview.sample_indices=[0,8,16]

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/preview_augmentation.yaml`.
    - Court augmentation has no per-transform `enabled` flags; the "augmented"
      panels are produced by building the dataset with `is_train=True` (full
      training pipeline) while the "original" panel uses `is_train=False`
      (deterministic validation resize only).
    - preview.require_pose must match the intended loss route. When true, the
      pipeline uses the same aspect-preserving camera-pose geometry as training.
    - Each row is one exact dataset draw. Columns separately expose RGB, the
      actual KP heatmap, categorical SEG masks, and binary LINE mask passed to
      the losses; no target is hidden beneath another target's overlay.
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
    denormalize_tensor_to_rgb,
)
from src.tasks.court_detection.visualization.rendering.target_preview import (
    render_heatmap_target,
    render_line_target,
    render_segmentation_target,
    summarize_targets,
)
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json

_BOUNDARY = "court_detection.preview_augmentation"


def _runtime(cfg: DictConfig) -> tuple[Path, CourtDataConfig]:
    root, resolver = validate_paths_boundary(cfg, expected_sections={"data", "preview"})
    data = CourtDataConfig.from_mapping(
        require_config_mapping(root, "data", path="configuration"), resolver=resolver
    )
    preview = require_config_mapping(root, "preview", path="configuration")
    expected = {
        "split",
        "require_pose",
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
    require_config_value(preview, "require_pose", bool, path="preview")
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
    if set(draw) != {"mask_alpha", "heatmap_alpha"}:
        raise ValueError("preview.draw has an invalid field set.")
    if set(layout) != {
        "tile_gap",
        "header_height",
        "text_scale",
        "text_thickness",
        "background_rgb",
    }:
        raise ValueError("preview.layout has an invalid field set.")
    require_config_value(draw, "mask_alpha", (float, int), path="preview.draw")
    require_config_value(draw, "heatmap_alpha", (float, int), path="preview.draw")
    for key in ("tile_gap", "header_height", "text_thickness"):
        require_config_value(layout, key, int, path="preview.layout")
    require_config_value(layout, "text_scale", (float, int), path="preview.layout")
    require_config_value(layout, "background_rgb", list, path="preview.layout")
    background = cast("list[object]", layout["background_rgb"])
    if len(background) != 3 or any(
        type(channel) is not int or not 0 <= channel <= 255 for channel in background
    ):
        raise ValueError("preview.layout.background_rgb must be three RGB integers.")
    mask_alpha = float(cast("float | int", draw["mask_alpha"]))
    heatmap_alpha = float(cast("float | int", draw["heatmap_alpha"]))
    if not 0.0 <= mask_alpha <= 1.0 or not 0.0 <= heatmap_alpha <= 1.0:
        raise ValueError("preview draw alpha values must be in [0, 1].")
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
    require_pose = bool(cfg.preview.require_pose)
    base_dataset = _dataset(
        data, split=split_name, is_train=False, require_pose=require_pose
    )
    augmented_dataset = _dataset(
        data, split=split_name, is_train=True, require_pose=require_pose
    )

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
        samples = [("orig", base_sample)]
        for variant in range(num_augmented):
            _seed_all(seed + sample_index * 1009 + variant + 1)
            augmented_sample = augmented_dataset[sample_index]
            samples.append((f"aug{variant}", augmented_sample))

        rendered_rows: list[tuple[list[np.ndarray], list[str]]] = []
        variant_metadata: list[dict[str, object]] = []
        sigma_ratio = _configured_sigma_ratio(data)
        for title, sample in samples:
            panels, panel_titles = _target_panels(
                sample,
                target_kinds=target_kinds,
                title=title,
                cfg=cfg,
            )
            rendered_rows.append((panels, panel_titles))
            variant_metadata.append(
                {
                    "variant": title,
                    "targets": summarize_targets(
                        cast("Mapping[str, object]", sample["targets"]),
                        sigma_ratio=sigma_ratio,
                    ),
                }
            )
        flattened_panels = [
            panel for panels, _titles in rendered_rows for panel in panels
        ]
        padded = _pad_panels_to_common_size(flattened_panels, cfg)
        rows: list[np.ndarray] = []
        cursor = 0
        for panels, panel_titles in rendered_rows:
            row_panels = padded[cursor : cursor + len(panels)]
            cursor += len(panels)
            rows.append(compose_titled_row(row_panels, panel_titles, cfg))
        sheet = _stack_rows(rows, cfg)

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
            "require_pose": require_pose,
            "output_image": str(image_path),
            "variants": variant_metadata,
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
    require_pose: bool,
) -> CourtDetectionDataset:
    if split not in {"train", "val"}:
        raise ValueError("Preview split must be train or val.")
    pipeline = build_court_processing_pipeline(
        data,
        is_train=is_train,
        require_pose=require_pose,
    )
    records = pipeline.input_layer.records(cast("CourtSourceSplit", split))
    return CourtDetectionDataset(records, pipeline=pipeline)


def _target_panels(
    sample: dict[str, Any],
    *,
    target_kinds: tuple[str, ...],
    title: str,
    cfg: DictConfig,
) -> tuple[list[np.ndarray], list[str]]:
    """Render RGB plus one non-overlapping panel per configured loss target."""
    rgb = denormalize_tensor_to_rgb(cast("torch.Tensor", sample["image"]))
    targets = cast("Mapping[str, object]", sample["targets"])
    panels = [rgb]
    titles = [f"{title}: RGB"]
    for kind in target_kinds:
        value = targets[kind]
        if kind == "kp":
            heatmap = cast("Mapping[str, torch.Tensor]", value)["heatmap"]
            panel = render_heatmap_target(
                rgb,
                heatmap,
                alpha=float(cfg.preview.draw.heatmap_alpha),
            )
            label = "KP heatmap (max)"
        elif kind == "seg":
            panel = render_segmentation_target(
                rgb,
                cast("torch.Tensor", value),
                alpha=float(cfg.preview.draw.mask_alpha),
            )
            label = "SEG classes 1..6"
        elif kind == "semantic_line":
            panel = render_segmentation_target(
                rgb,
                cast("torch.Tensor", value),
                alpha=float(cfg.preview.draw.mask_alpha),
                max_label=11,
            )
            label = "SEMANTIC LINE classes 1..11"
        elif kind == "line":
            panel = render_line_target(
                rgb,
                cast("torch.Tensor", value),
                alpha=float(cfg.preview.draw.mask_alpha),
            )
            label = "LINE binary"
        else:  # pragma: no cover - strict configuration owns target kinds
            raise ValueError(f"Unknown Court target: {kind!r}")
        panels.append(panel)
        titles.append(f"{title}: {label}")
    return panels, titles


def _configured_sigma_ratio(data: CourtDataConfig) -> float | None:
    for target in data.processing.targets:
        if target.kind == "kp":
            if target.sigma_ratio is None:  # pragma: no cover - typed config owns it
                raise ValueError("Configured KP target has no sigma_ratio.")
            return float(target.sigma_ratio)
    return None


def _stack_rows(rows: list[np.ndarray], cfg: DictConfig) -> np.ndarray:
    """Pad and stack variant rows without rescaling any target geometry."""
    gap = int(cfg.preview.layout.tile_gap)
    background = tuple(int(value) for value in cfg.preview.layout.background_rgb)
    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + gap * (len(rows) - 1)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    cursor = 0
    for row in rows:
        canvas[cursor : cursor + row.shape[0], : row.shape[1]] = row
        cursor += row.shape[0] + gap
    return cast("np.ndarray", canvas)


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
