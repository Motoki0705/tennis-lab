"""
Sweep the bbox max-side ratio of the web ball-detection sources and export
candidate "ball-too-close" frames for visual threshold selection.

The per-frame metric is ``ratio = max over boxes of max(w / W, h / H)`` where
``w, h`` is a ball bbox and ``W, H`` the frame size. Frames whose ratio is large
show the ball zoomed in close, a domain that differs from the rally-camera
detection task and is a candidate for cleaning. Only the bbox-bearing sources
(Roboflow COCO and Ball-YOLO) are analyzed; the temporal point-only sources
(racketvision, kaggle) carry no bbox and are skipped by construction.

Outputs (under ``analyze.output_dir``):
    - ``frame_ratios.csv``: one row per frame with its ratio + locator, reusable
      by the later cleaning step.
    - ``sweep.csv`` / ``sweep.md``: removal counts per threshold, overall and
      per source.
    - ``histogram.csv``: ratio histogram.
    - ``samples/<source>/bin_<lo>-<hi>/*.jpg``: annotated example frames per
      ratio bin for eyeballing.

Usage:
    python -m src.tasks.ball_detection.scripts.analyze_web_bbox_ratio
    python -m src.tasks.ball_detection.scripts.analyze_web_bbox_ratio \
        analyze.samples_per_bin=64 analyze.sources.ball_yolo=false

Notes:
    - Hydra config: ``src/tasks/ball_detection/configs/analyze_web_bbox_ratio.yaml``.
    - No frames are deleted; this script only reports and exports samples.
    - Ratios are computed from raw annotations, independent of the unified store
      (which keeps ball centers only, not bbox sizes).
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.geometry.bbox import bbox_max_side_ratio
from src.utils.hydra import hydra_main
from src.utils.io import ensure_dir, load_json
from src.utils.video.encoding import iter_selected_video_jpegs

_ROBOFLOW_DATASETS = (
    "roboflow_tennis_ball_tracking_detection_h9rat_v1",
    "roboflow_tennis_ball_tracking_1wnxz_v2",
    "roboflow_tennis_ball_wafqb_v2",
)
_RAW_SPLITS = ("train", "valid", "test")


@dataclass
class FrameRatio:
    """One frame, its dominant-bbox ratio, and how to load it for export."""

    source: str
    ratio: float
    n_boxes: int
    # Normalized [0,1] boxes (x1, y1, x2, y2) for drawing at export time.
    norm_boxes: list[tuple[float, float, float, float]]
    # Locators (exactly one applies per source).
    image_path: str | None = None
    video_key: tuple[str, ...] | None = None
    frame_index: int = -1


def _roboflow_frames(web_root: Path) -> Iterator[FrameRatio]:
    """Yield per-frame ratios for all configured Roboflow COCO exports."""
    for name in _ROBOFLOW_DATASETS:
        dataset_dir = web_root / name
        for raw_split in _RAW_SPLITS:
            split_dir = dataset_dir / raw_split
            annotations = split_dir / "_annotations.coco.json"
            if not annotations.exists():
                continue
            coco = load_json(annotations)
            ball_categories = {
                category["id"]
                for category in coco["categories"]
                if str(category.get("supercategory", "none")).lower() != "none"
            }
            boxes_by_image: dict[int, list[list[float]]] = defaultdict(list)
            for ann in coco["annotations"]:
                if ann["category_id"] not in ball_categories:
                    continue
                boxes_by_image[ann["image_id"]].append([float(v) for v in ann["bbox"]])
            for image in coco["images"]:
                boxes = boxes_by_image.get(image["id"], [])
                if not boxes:
                    continue
                w_img = float(image["width"])
                h_img = float(image["height"])
                norm_boxes: list[tuple[float, float, float, float]] = []
                ratio = 0.0
                for x, y, bw, bh in boxes:
                    ratio = max(ratio, bbox_max_side_ratio(bw, bh, w_img, h_img))
                    norm_boxes.append(
                        (x / w_img, y / h_img, (x + bw) / w_img, (y + bh) / h_img)
                    )
                yield FrameRatio(
                    source=name,
                    ratio=ratio,
                    n_boxes=len(boxes),
                    norm_boxes=norm_boxes,
                    image_path=str(split_dir / image["file_name"]),
                )


def _ball_yolo_frames(web_root: Path) -> Iterator[FrameRatio]:
    """Yield per-frame ratios for the Ball-YOLO labels mapped to source videos."""
    labels_root = web_root / "ball_yolo_sport_ball_labels" / "tennis" / "Labels"
    videos_dir = web_root / "sport_ball_detection_videos" / "tennis" / "Videos"
    mapping_csv = web_root / "ball_yolo_tennis_video_mapping.csv"
    if not mapping_csv.exists():
        return
    with mapping_csv.open(encoding="utf-8") as handle:
        mapping = {row["label_folder"]: row for row in csv.DictReader(handle)}
    for folder in sorted(labels_root.iterdir()):
        if not folder.is_dir() or folder.name not in mapping:
            continue
        parts = [
            videos_dir / part
            for part in mapping[folder.name]["official_video_files"].split(";")
        ]
        parts = [part for part in parts if part.exists()]
        if not parts:
            continue
        video_key = tuple(str(part) for part in parts)
        for label_file in folder.glob("*.txt"):
            frame_index = int(label_file.stem.rsplit("_", 1)[1])
            norm_boxes: list[tuple[float, float, float, float]] = []
            ratio = 0.0
            for line in label_file.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in fields[1:5])
                ratio = max(ratio, bbox_max_side_ratio(bw, bh, 1.0, 1.0))
                norm_boxes.append(
                    (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)
                )
            if not norm_boxes:
                continue
            yield FrameRatio(
                source="ball_yolo",
                ratio=ratio,
                n_boxes=len(norm_boxes),
                norm_boxes=norm_boxes,
                video_key=video_key,
                frame_index=frame_index,
            )


def _collect(cfg: DictConfig, web_root: Path) -> list[FrameRatio]:
    """Run the enabled source readers and collect all frame ratios."""
    frames: list[FrameRatio] = []
    if bool(cfg.sources.roboflow):
        for frame in tqdm(_roboflow_frames(web_root), desc="roboflow"):
            frames.append(frame)
    if bool(cfg.sources.ball_yolo):
        for frame in tqdm(_ball_yolo_frames(web_root), desc="ball_yolo"):
            frames.append(frame)
    return frames


def _write_frame_ratios(path: Path, frames: list[FrameRatio]) -> None:
    """Persist the per-frame ratio table (reusable by the cleaning step)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "ratio", "n_boxes", "image_path", "video_key", "frame_index"]
        )
        for frame in frames:
            writer.writerow(
                [
                    frame.source,
                    f"{frame.ratio:.6f}",
                    frame.n_boxes,
                    frame.image_path or "",
                    ";".join(frame.video_key) if frame.video_key else "",
                    frame.frame_index,
                ]
            )


def _write_sweep(
    csv_path: Path,
    md_path: Path,
    frames: list[FrameRatio],
    thresholds: list[float],
) -> list[tuple[float, int, float]]:
    """Write removal counts per threshold, overall and per source."""
    sources = sorted({frame.source for frame in frames})
    by_source: dict[str, np.ndarray] = {
        source: np.array(
            [frame.ratio for frame in frames if frame.source == source],
            dtype=np.float64,
        )
        for source in sources
    }
    all_ratios = np.array([frame.ratio for frame in frames], dtype=np.float64)
    total = len(all_ratios)

    rows: list[list[Any]] = []
    overall: list[tuple[float, int, float]] = []
    for threshold in thresholds:
        removed = int((all_ratios >= threshold).sum())
        pct = 100.0 * removed / total if total else 0.0
        overall.append((threshold, removed, pct))
        row: list[Any] = [f"{threshold:.2f}", removed, f"{pct:.2f}"]
        for source in sources:
            ratios = by_source[source]
            src_removed = int((ratios >= threshold).sum())
            src_pct = 100.0 * src_removed / len(ratios) if len(ratios) else 0.0
            row += [src_removed, f"{src_pct:.2f}"]
        rows.append(row)

    header = ["threshold", "removed", "removed_pct"]
    for source in sources:
        header += [f"{source}_removed", f"{source}_pct"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    lines = [
        f"# Web bbox max-side ratio sweep (total frames with boxes: {total})",
        "",
        "ratio = max over boxes of max(w/W, h/H); a frame is *removed* if ratio >= threshold.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall


def _write_histogram(path: Path, frames: list[FrameRatio]) -> None:
    """Write a fixed-width ratio histogram."""
    ratios = np.array([frame.ratio for frame in frames], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 51)
    counts, _ = np.histogram(np.clip(ratios, 0.0, 1.0), bins=edges)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_lo", "bin_hi", "count"])
        for lo, hi, count in zip(edges[:-1], edges[1:], counts, strict=False):
            writer.writerow([f"{lo:.3f}", f"{hi:.3f}", int(count)])


def _draw(image: np.ndarray, frame: FrameRatio) -> np.ndarray:
    """Draw normalized boxes and the ratio label on a copy of the image."""
    out = image.copy()
    h, w = out.shape[:2]
    for nx1, ny1, nx2, ny2 in frame.norm_boxes:
        p1 = (int(round(nx1 * w)), int(round(ny1 * h)))
        p2 = (int(round(nx2 * w)), int(round(ny2 * h)))
        cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
    cv2.putText(
        out,
        f"ratio={frame.ratio:.3f} n={frame.n_boxes}",
        (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def _bin_label(edges: list[float], ratio: float) -> tuple[int, str]:
    """Return the bin index and a human label for a ratio."""
    for index in range(len(edges) - 1):
        if edges[index] <= ratio < edges[index + 1]:
            return index, f"bin_{edges[index]:.2f}-{edges[index + 1]:.2f}"
    last = len(edges) - 2
    return last, f"bin_{edges[last]:.2f}-{edges[last + 1]:.2f}"


def _export_samples(cfg: DictConfig, frames: list[FrameRatio], out_root: Path) -> None:
    """Export up to ``samples_per_bin`` annotated frames per (source, ratio bin)."""
    edges = [float(value) for value in cfg.export_bin_edges]
    per_bin = int(cfg.samples_per_bin)
    rng = random.Random(int(cfg.seed))

    buckets: dict[tuple[str, str], list[FrameRatio]] = defaultdict(list)
    for frame in frames:
        _, label = _bin_label(edges, frame.ratio)
        buckets[(frame.source, label)].append(frame)

    selected: list[FrameRatio] = []
    bucket_dir: dict[int, Path] = {}
    for (source, label), members in buckets.items():
        rng.shuffle(members)
        chosen = members[:per_bin]
        target = out_root / source / label
        ensure_dir(target)
        for frame in chosen:
            bucket_dir[id(frame)] = target
            selected.append(frame)

    # Roboflow: still images on disk.
    for frame in tqdm(
        [f for f in selected if f.image_path], desc="export-roboflow"
    ):
        image = cv2.imread(frame.image_path)
        if image is None:
            continue
        target = bucket_dir[id(frame)]
        name = Path(cast(str, frame.image_path)).stem
        cv2.imwrite(str(target / f"{name}.jpg"), _draw(image, frame))

    # Ball-YOLO: decode only the sampled frames, grouped per video.
    by_video: dict[tuple[str, ...], list[FrameRatio]] = defaultdict(list)
    for frame in selected:
        if frame.video_key is not None:
            by_video[frame.video_key].append(frame)
    for video_key, members in tqdm(by_video.items(), desc="export-ball_yolo"):
        wanted = {frame.frame_index: frame for frame in members}
        for index, jpeg in iter_selected_video_jpegs(
            list(video_key), set(wanted), quality=int(cfg.jpeg_quality)
        ):
            frame = wanted[index]
            buffer = np.frombuffer(jpeg, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                continue
            target = bucket_dir[id(frame)]
            stem = Path(video_key[0]).stem
            cv2.imwrite(str(target / f"{stem}_f{index:06d}.jpg"), _draw(image, frame))


@hydra_main(
    config_path="../configs",
    config_name="analyze_web_bbox_ratio",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Collect ratios, write the sweep tables, and export sample frames."""
    analyze = cfg.analyze
    web_root = Path(to_absolute_path(str(analyze.web_root)))
    out_root = Path(to_absolute_path(str(analyze.output_dir)))
    ensure_dir(out_root)

    frames = _collect(analyze, web_root)
    if not frames:
        print("[analyze_web_bbox_ratio] no bbox-bearing frames found.")
        return 1
    print(f"[analyze_web_bbox_ratio] collected {len(frames)} frames with boxes.")

    _write_frame_ratios(out_root / "frame_ratios.csv", frames)
    overall = _write_sweep(
        out_root / "sweep.csv",
        out_root / "sweep.md",
        frames,
        [float(value) for value in analyze.sweep_thresholds],
    )
    _write_histogram(out_root / "histogram.csv", frames)

    print("[analyze_web_bbox_ratio] removal sweep (threshold: removed frames, %):")
    for threshold, removed, pct in overall:
        print(f"    >= {threshold:.2f}: {removed:6d}  ({pct:5.2f}%)")

    _export_samples(analyze, frames, out_root / "samples")
    print(f"[analyze_web_bbox_ratio] wrote outputs under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
