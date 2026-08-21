"""Render qualitative court-keypoint heatmap previews for multiple sigma ratios.

Usage:
    python -m src.tasks.court_detection.scripts.preview_heatmaps
    python -m src.tasks.court_detection.scripts.preview_heatmaps preview.ratios=[0.004,0.008,0.016]
    python -m src.tasks.court_detection.scripts.preview_heatmaps preview.sample_indices=[10,20] preview.split=val

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/preview_heatmaps.yaml`.
    - The configured source adapter supplies canonical point groups and visibility.
    - Both ordered KP14 and multi-peak KP7 channels use the shared heatmap utility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.base.visualization.preview import (
    compose_titled_row as _compose_row,
)
from src.tasks.base.visualization.preview import (
    draw_normalized_point as _draw_point,
)
from src.tasks.base.visualization.preview import (
    resolve_sample_indices,
)
from src.tasks.court_detection.configuration import (
    CourtDataConfig,
    validate_paths_boundary,
)
from src.tasks.court_detection.data.contracts import CourtSourceSplit
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.utils.configuration import PathRole
from src.utils.data.heatmaps import generate_gaussian_heatmaps, heatmaps_to_argmax
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json

_BOUNDARY = "court_detection.preview_heatmaps"


def _runtime(cfg: DictConfig) -> tuple[Path, CourtDataConfig]:
    root, resolver = validate_paths_boundary(cfg, expected_sections={"data", "preview"})
    data = CourtDataConfig.from_mapping(
        require_config_mapping(root, "data", path="configuration"), resolver=resolver
    )
    if "kp" not in {target.kind for target in data.processing.targets}:
        raise ValueError("preview_heatmaps requires kp in data.processing.targets.")
    preview = require_config_mapping(root, "preview", path="configuration")
    expected = {
        "split",
        "sample_indices",
        "max_samples",
        "ratios",
        "output_dir",
        "layout",
        "draw",
    }
    if set(preview) != expected:
        raise ValueError(f"preview requires exactly {sorted(expected)}.")
    ratios = cast(
        "list[object] | tuple[object, ...]",
        require_config_value(preview, "ratios", (list, tuple), path="preview"),
    )
    if not all(
        type(value) in (float, int) and float(cast("float | int", value)) > 0
        for value in ratios
    ):
        raise ValueError("preview.ratios must contain positive numbers.")
    for key in ("split", "output_dir"):
        require_config_value(preview, key, str, path="preview")
    sample_indices = cast(
        "list[object]",
        require_config_value(preview, "sample_indices", list, path="preview"),
    )
    if any(type(index) is not int or index < 0 for index in sample_indices):
        raise ValueError("preview.sample_indices must contain non-negative integers.")
    max_samples = cast(
        "int", require_config_value(preview, "max_samples", int, path="preview")
    )
    if max_samples <= 0:
        raise ValueError("preview.max_samples must be positive.")
    if cast("str", preview["split"]) not in {"train", "val"}:
        raise ValueError("preview.split must be train or val.")
    layout = require_config_mapping(preview, "layout", path="preview")
    draw = require_config_mapping(preview, "draw", path="preview")
    if set(layout) != {
        "tile_gap",
        "header_height",
        "text_scale",
        "text_thickness",
        "background_rgb",
    }:
        raise ValueError("preview.layout has an invalid field set.")
    if set(draw) != {"gt_radius", "argmax_radius", "thickness"}:
        raise ValueError("preview.draw has an invalid field set.")
    for key in ("tile_gap", "header_height", "text_thickness"):
        require_config_value(layout, key, int, path="preview.layout")
    require_config_value(layout, "text_scale", (float, int), path="preview.layout")
    require_config_value(layout, "background_rgb", list, path="preview.layout")
    for key in ("gt_radius", "argmax_radius", "thickness"):
        require_config_value(draw, key, int, path="preview.draw")
    background = cast("list[object]", layout["background_rgb"])
    if len(background) != 3 or any(
        type(channel) is not int or not 0 <= channel <= 255 for channel in background
    ):
        raise ValueError("preview.layout.background_rgb must be three RGB integers.")
    if any(cast("int", layout[key]) < 0 for key in ("tile_gap",)) or any(
        cast("int", layout[key]) <= 0 for key in ("header_height", "text_thickness")
    ):
        raise ValueError("preview.layout sizes are invalid.")
    if float(cast("float | int", layout["text_scale"])) <= 0 or any(
        cast("int", draw[key]) <= 0
        for key in ("gt_radius", "argmax_radius", "thickness")
    ):
        raise ValueError("preview draw/text sizes must be positive.")
    output_dir = cast("str", preview["output_dir"])
    if not output_dir:
        raise ValueError("preview.output_dir must not be empty.")
    return resolver.resolve(PathRole.OUTPUT, output_dir), data


def _validate_boundary(cfg: DictConfig) -> None:
    _runtime(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="preview_heatmaps",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    output_dir, data = _runtime(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = CourtDerivedTargetStore(data.processing.derived_target_root)
    input_layer = build_court_input(data.source, target_store=store)
    records = input_layer.records(cast("CourtSourceSplit", str(cfg.preview.split)))
    sample_indices = resolve_sample_indices(cfg, len(records))

    manifest: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        raw = input_layer.load(records[sample_index])
        image_id = raw.sample_id
        image = np.asarray(raw.image, dtype=np.uint8)
        height, width = image.shape[:2]
        channels = raw.keypoint_channels
        if channels is None:
            raise ValueError(f"Court sample {image_id!r} has no keypoint channels.")
        scale = channels.points_xy.new_tensor(
            [float(max(width - 1, 1)), float(max(height - 1, 1))]
        )
        centers = channels.points_xy / scale
        visible = channels.point_visible
        centers_xy = centers[visible].cpu().numpy()

        panels = [_annotate_original(image, centers_xy, cfg)]
        ratios = [float(value) for value in cfg.preview.ratios]
        ratio_records: list[dict[str, Any]] = []
        for sigma_ratio in ratios:
            heatmaps = generate_gaussian_heatmaps(
                size_hw=(height, width),
                centers_xy=centers,
                sigma_ratio=sigma_ratio,
                visibility=visible,
                point_reduction="max",
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

        canvas = _compose_row(
            panels, ["original", *[f"ratio={ratio:.4f}" for ratio in ratios]], cfg
        )
        safe_id = image_id.replace(":", "_")
        image_path = output_dir / f"sample_{sample_index:05d}_{safe_id}.png"
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


def _annotate_original(
    image: np.ndarray, centers_xy: np.ndarray, cfg: DictConfig
) -> np.ndarray:
    canvas = image.copy()
    for center_xy in centers_xy:
        _draw_point(
            canvas,
            cast("tuple[float, float]", tuple(float(v) for v in center_xy.tolist())),
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
    heatmaps_np = np.asarray(
        heatmaps.detach().cpu().numpy() if hasattr(heatmaps, "detach") else heatmaps
    )
    pooled = np.clip(heatmaps_np.max(axis=0), 0.0, 1.0)
    heatmap_cm = cv2.applyColorMap((pooled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_cm = cv2.cvtColor(heatmap_cm, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.55, heatmap_cm, 0.45, 0.0)

    for center_xy in gt_centers_xy:
        _draw_point(
            overlay,
            cast("tuple[float, float]", tuple(float(v) for v in center_xy.tolist())),
            radius=int(cfg.preview.draw.gt_radius),
            color=(255, 80, 80),
            thickness=1,
        )
    for center_xy, peak_value in zip(argmax_xy, peak_values, strict=True):
        if float(peak_value) <= 0:
            continue
        _draw_point(
            overlay,
            cast("tuple[float, float]", tuple(float(v) for v in center_xy.tolist())),
            radius=int(cfg.preview.draw.argmax_radius),
            color=(80, 255, 120),
            thickness=int(cfg.preview.draw.thickness),
        )
    return cast("np.ndarray", overlay)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
