"""Create label, heatmap, and metric diagnostics for a court release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from src.synthetic_data_generation.configuration import (
    add_path_roots_argument,
    non_hydra_path_resolver,
)
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    _canonical_sha256,
    _relative_file_ref,
    _sha256_file,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    SYMMETRIC_KEYPOINT_CLASS_NAMES,
    decode_heatmap_atlas_u16,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import DATASET_SCHEMA
from src.synthetic_data_generation.dataset.court.rendering.orbit_preview import (
    _CLASS_COLOURS,
    _draw_marker,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DIAGNOSTIC_SCHEMA = "tennis_multicourt_dataset_diagnostic_v1"
_BUCKETS = ("full", "near_full", "partial", "sparse")
_SPLITS = ("train", "validation", "test")
PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.court.dataset_preview",
    fields=(
        BoundaryPathField(
            "dataset_dir",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "output_dir", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def _load_dataset(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema") != DATASET_SCHEMA:
        raise RuntimeError("Unsupported court dataset schema.")
    declared = payload.get("dataset_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("dataset_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise RuntimeError("Court dataset fingerprint mismatch.")
    return cast(dict[str, Any], payload)


def _select_grid(
    root: Path,
    dataset: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for frame in dataset["frames"]:
        labels = json.loads(
            (root / frame["labels"]["relative_path"]).read_text(encoding="utf-8")
        )
        split = frame["split"]
        buckets = labels["geometric_coverage"]
        for bucket in _BUCKETS:
            if bucket in buckets and (split, bucket) not in selected:
                selected[(split, bucket)] = (frame, labels)
    missing = [
        (split, bucket)
        for bucket in _BUCKETS
        for split in _SPLITS
        if (split, bucket) not in selected
    ]
    if missing:
        raise RuntimeError(f"Dataset lacks diagnostic split/bucket cells: {missing}")
    return [selected[(split, bucket)] for bucket in _BUCKETS for split in _SPLITS]


def _label_sheet(
    root: Path,
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    output: Path,
) -> None:
    columns = len(_SPLITS)
    rows = len(_BUCKETS)
    width, height = selected[0][1]["resolution"]
    caption_height = 48
    sheet = Image.new(
        "RGB",
        (columns * width, rows * (height + caption_height)),
        (24, 24, 24),
    )
    for index, (frame, labels) in enumerate(selected):
        image = Image.open(root / frame["rgb"]["relative_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for court_index, court in enumerate(labels["projection"]["courts"]):
            for class_record in court["classes"]:
                colour = _CLASS_COLOURS[class_record["class_id"]]
                for point in class_record["points"]:
                    if point["in_frame"]:
                        _draw_marker(
                            draw,
                            xy=tuple(point["uv"]),
                            colour=colour,
                            court_index=court_index,
                            visible=point["visible"],
                        )
        column = index % columns
        row = index // columns
        x = column * width
        y = row * (height + caption_height)
        sheet.paste(image, (x, y))
        caption = (
            f"{frame['split']} / {_BUCKETS[row]}\n"
            f"{frame['family_id']} coverage={labels['geometric_coverage']} "
            f"visible={labels['renderer_visible_points']}"
        )
        ImageDraw.Draw(sheet).multiline_text(
            (x + 4, y + height + 3),
            caption,
            fill=(240, 240, 240),
            spacing=2,
        )
    sheet.save(output)


def _heatmap_sheet(
    root: Path,
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    output: Path,
) -> None:
    columns = len(_SPLITS)
    rows = len(_BUCKETS)
    width, height = selected[0][1]["resolution"]
    sheet = Image.new("RGB", (columns * width, rows * height), (0, 0, 0))
    colours = np.asarray(_CLASS_COLOURS, dtype=np.float32) / 255.0
    for index, (frame, _) in enumerate(selected):
        rgb = (
            np.asarray(
                Image.open(root / frame["rgb"]["relative_path"]).convert("RGB"),
                dtype=np.float32,
            )
            / 255.0
        )
        atlas = np.asarray(Image.open(root / frame["heatmap_atlas"]["relative_path"]))
        heatmaps = decode_heatmap_atlas_u16(atlas)
        maximum = heatmaps.max(axis=0)
        weighted = np.einsum("chw,cd->hwd", heatmaps, colours)
        denominator = np.maximum(heatmaps.sum(axis=0, keepdims=False), 1.0e-6)
        heat_colour = weighted / denominator[..., None]
        alpha = np.clip(maximum[..., None] * 0.75, 0.0, 0.75)
        diagnostic = np.rint(
            255.0 * ((1.0 - alpha) * rgb + alpha * heat_colour)
        ).astype(np.uint8)
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(Image.fromarray(diagnostic, mode="RGB"), (x, y))
    sheet.save(output)


def _metric_plot(dataset: dict[str, Any], output: Path) -> None:
    metrics = dataset["metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(_BUCKETS))
    bar_width = 0.24
    for split_index, split in enumerate(_SPLITS):
        counts = metrics["split_coverage_counts"][split]
        axes[0].bar(
            x + (split_index - 1) * bar_width,
            [counts.get(bucket, 0) for bucket in _BUCKETS],
            width=bar_width,
            label=split,
        )
    axes[0].set(
        title="Court-instance coverage by family-disjoint split",
        ylabel="court-instance projections",
        xticks=x,
        xticklabels=_BUCKETS,
    )
    axes[0].legend()
    visible = metrics["renderer_visible_points_by_class"]
    names = list(SYMMETRIC_KEYPOINT_CLASS_NAMES)
    axes[1].barh(
        names,
        [visible[name] for name in names],
        color=[np.asarray(colour) / 255.0 for colour in _CLASS_COLOURS],
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1].set(
        title="Renderer-visible physical points by 7-class target",
        xlabel="visible points across 428 frames",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_path_roots_argument(parser)
    args = parser.parse_args()
    paths = PATH_BOUNDARY.validate(
        {"dataset_dir": args.dataset_dir, "output_dir": args.output_dir},
        resolver=non_hydra_path_resolver(args.path_roots),
    )
    dataset_root = paths.declared("dataset_dir").path
    output_dir = paths.declared("output_dir").path
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")

    dataset = _load_dataset(dataset_root)
    selected = _select_grid(dataset_root, dataset)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        label_path = temporary / "label-contact-sheet.png"
        heatmap_path = temporary / "heatmap-contact-sheet.png"
        metric_path = temporary / "split-and-class-metrics.png"
        _label_sheet(dataset_root, selected, label_path)
        _heatmap_sheet(dataset_root, selected, heatmap_path)
        _metric_plot(dataset, metric_path)
        manifest: dict[str, object] = {
            "schema": DIAGNOSTIC_SCHEMA,
            "dataset": {
                "manifest_sha256": _sha256_file(dataset_root / "manifest.json"),
                "dataset_fingerprint": dataset["dataset_fingerprint"],
            },
            "grid": {
                "columns": list(_SPLITS),
                "rows": list(_BUCKETS),
                "sample_camera_ids": [frame["camera_id"] for frame, _ in selected],
            },
            "files": {
                "label_contact_sheet": _relative_file_ref(temporary, label_path),
                "heatmap_contact_sheet": _relative_file_ref(
                    temporary,
                    heatmap_path,
                ),
                "split_and_class_metrics": _relative_file_ref(
                    temporary,
                    metric_path,
                ),
            },
            "rgb_overlay_scope": "diagnostic-only; dataset RGB is unchanged",
        }
        manifest["diagnostic_fingerprint"] = _canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
