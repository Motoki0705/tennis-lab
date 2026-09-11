"""Qualitative target audit for completed YouTube Court annotations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.data.target_generation.line import generate_line_target
from src.tasks.court_detection.data.target_generation.segmentation import (
    generate_segmentation_target,
)
from src.tasks.court_detection.visualization.rendering.target_preview import (
    gaussian_pixel_geometry,
    render_heatmap_target,
    render_line_target,
    render_segmentation_target,
)
from src.utils.configuration import PathResolver, PathRole
from src.utils.data.heatmaps import generate_gaussian_heatmaps
from src.utils.io import load_json, save_json_atomic


def write_youtube_target_previews(
    config: Mapping[str, object],
    *,
    resolver: PathResolver,
    dataset_root: Path,
    annotations_dir: Path,
    output_dir: Path,
) -> None:
    """Render configured candidate targets without mutating source annotations."""
    split = str(config["split"])
    annotation_path = resolver.resolve_beneath(
        PathRole.DATA, annotations_dir, f"{split}.json"
    )
    payload = load_json(annotation_path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
        raise ValueError(f"YouTube annotation payload is invalid: {annotation_path}")
    items = cast("list[object]", payload["items"])
    completed = [
        cast("Mapping[str, object]", item)
        for item in items
        if isinstance(item, Mapping) and item.get("annotation_status") == "completed"
    ]
    ready = [item for item in completed if ground_kp14_ready(item)]
    if not ready:
        raise ValueError(
            f"No completed {split} annotation contains finite visible ground KP14."
        )
    explicit = [int(value) for value in cast("Sequence[int]", config["sample_indices"])]
    indices = explicit or list(
        range(min(cast("int", config["max_samples"]), len(ready)))
    )
    if any(index >= len(ready) for index in indices):
        raise IndexError(
            f"YouTube target preview index exceeds ready sample count {len(ready)}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _render_one(
            ready[index],
            sample_index=index,
            config=config,
            resolver=resolver,
            dataset_root=dataset_root,
            output_dir=output_dir,
        )
        for index in indices
    ]
    save_json_atomic(
        {
            "split": split,
            "completed_annotations": len(completed),
            "ground_kp14_ready": len(ready),
            "not_ready": len(completed) - len(ready),
            "samples": records,
        },
        output_dir / "manifest.json",
    )
    print(
        "[prepare_youtube_dataset] target preview "
        f"samples={len(records)} ready={len(ready)}/{len(completed)} -> {output_dir}"
    )


def ground_kp14_ready(item: Mapping[str, object]) -> bool:
    """Return whether one completed annotation can define all dense targets."""
    points = item.get("keypoints")
    if not isinstance(points, list) or len(points) < 14:
        return False
    for raw in points[:14]:
        if not isinstance(raw, Mapping) or raw.get("visibility") not in {1, 2}:
            return False
        if any(not _finite_coordinate(raw.get(axis)) for axis in ("x", "y")):
            return False
    return True


def _render_one(
    item: Mapping[str, object],
    *,
    sample_index: int,
    config: Mapping[str, object],
    resolver: PathResolver,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    sample_id = str(item["id"])
    image_path = resolver.resolve_beneath(
        PathRole.DATA, dataset_root, str(item["image_path"])
    )
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"YouTube preview image is unreadable: {image_path}")
    rgb = cast(
        "NDArray[np.uint8]",
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False),
    )
    height, width = rgb.shape[:2]
    raw_points = cast("list[Mapping[str, object]]", item["keypoints"])
    points = torch.tensor(
        [
            [_coordinate(point, "x"), _coordinate(point, "y")]
            for point in raw_points[:14]
        ],
        dtype=torch.float32,
    )
    scale = points.new_tensor([float(max(width - 1, 1)), float(max(height - 1, 1))])
    normalized = points[:, None, :] / scale
    visible = torch.ones(14, 1, dtype=torch.bool)
    instance = CourtInstance2D(
        court_instance_id=f"{sample_id}:court",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=points,
        point_in_front=torch.ones(14, dtype=torch.bool),
        point_visible=torch.ones(14, dtype=torch.bool),
    )
    display_width = cast("int", config["display_width"])
    annotated = rgb.copy()
    for point in points:
        cv2.circle(
            annotated,
            tuple(int(round(float(value))) for value in point.tolist()),
            5,
            (255, 80, 80),
            2,
            lineType=cv2.LINE_AA,
        )
    panels = [_resize(annotated, display_width)]
    titles = ["annotated RGB | ground KP14"]
    sigma_records: list[dict[str, float]] = []
    for raw_ratio in cast("Sequence[float]", config["sigma_ratios"]):
        ratio = float(raw_ratio)
        heatmaps = generate_gaussian_heatmaps(
            (height, width),
            normalized,
            ratio,
            visibility=visible,
            point_reduction="max",
        )
        geometry = gaussian_pixel_geometry(ratio, height=height, width=width)
        panels.append(
            _resize(
                render_heatmap_target(
                    rgb,
                    heatmaps,
                    alpha=float(cast("float | int", config["heatmap_alpha"])),
                ),
                display_width,
            )
        )
        titles.append(f"KP sigma={ratio:.4f} | sigma={geometry['sigma_px']:.1f}px")
        sigma_records.append(geometry)
    seg = generate_segmentation_target(
        height=height, width=width, instances=(instance,)
    )
    panels.append(
        _resize(
            render_segmentation_target(
                rgb,
                seg,
                alpha=float(cast("float | int", config["mask_alpha"])),
            ),
            display_width,
        )
    )
    titles.append("SEG | 7 classes")
    line_records: list[dict[str, float | int]] = []
    multiplier = float(cast("float | int", config["baseline_width_multiplier"]))
    for raw_width in cast("Sequence[float]", config["line_width_metres"]):
        line_width = float(raw_width)
        baseline_width = line_width * multiplier
        line = generate_line_target(
            height=height,
            width=width,
            instances=(instance,),
            line_width_metres=line_width,
            baseline_width_metres=baseline_width,
        )
        panels.append(
            _resize(
                render_line_target(
                    rgb,
                    line,
                    alpha=float(cast("float | int", config["mask_alpha"])),
                ),
                display_width,
            )
        )
        titles.append(
            f"LINE {line_width * 100:.1f}cm | baseline {baseline_width * 100:.1f}cm"
        )
        line_records.append(
            {
                "line_width_metres": line_width,
                "baseline_width_metres": baseline_width,
                "foreground_pixels": int(np.count_nonzero(line)),
            }
        )
    sheet = _compose_row(panels, titles)
    preview_path = output_dir / f"{sample_index:05d}_{sample_id}.jpg"
    if not cv2.imwrite(
        str(preview_path),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    ):
        raise RuntimeError(f"Failed to write YouTube target preview: {preview_path}")
    return {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "source_image": str(image_path),
        "source_shape": [height, width, 3],
        "output_image": str(preview_path),
        "kp": sigma_records,
        "seg": {
            "shape": list(seg.shape),
            "class_pixel_counts": {
                str(label): int(np.count_nonzero(seg == label)) for label in range(7)
            },
        },
        "line": line_records,
    }


def _finite_coordinate(value: object) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _coordinate(point: Mapping[str, object], key: str) -> float:
    value = point.get(key)
    if not _finite_coordinate(value):
        raise ValueError(f"YouTube preview keypoint {key} must be finite.")
    return float(cast("float | int", value))


def _resize(rgb: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    height = max(1, int(round(rgb.shape[0] * width / rgb.shape[1])))
    return cast(
        "NDArray[np.uint8]",
        cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA),
    )


def _compose_row(
    panels: list[NDArray[np.uint8]], titles: list[str]
) -> NDArray[np.uint8]:
    gap, header = 10, 34
    height, width = panels[0].shape[:2]
    canvas = np.full(
        (height + header, len(panels) * width + gap * (len(panels) - 1), 3),
        24,
        dtype=np.uint8,
    )
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        if panel.shape[:2] != (height, width):
            raise ValueError("YouTube target preview panels must share dimensions.")
        left = index * (width + gap)
        canvas[header:, left : left + width] = panel
        cv2.putText(
            canvas,
            title,
            (left + 6, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


__all__ = ["ground_kp14_ready", "write_youtube_target_previews"]
