"""Overview:
Build a binary review queue and overlay images from Grounding DINO raw pseudo-label predictions.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/build_review_queue.py review_unit=image
    .venv/bin/python experiments/dino_lora_workflow/scripts/build_review_queue.py review_unit=annotation min_score=0.35 allowed_tasks='[tennis_roles]'

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/build_review_queue.yaml`.
    - Use `review_unit=image` for CourtKP20 reviews where all keypoints on an image are accepted or rejected together.
    - Use `review_unit=annotation` for tennis roles, ball boys, umpires, and line judges where each box is reviewed independently.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger("dino_review_queue")
IMAGE_KEYS = (
    "absolute_image_path",
    "image_path",
    "image",
    "output_file",
    "absolute_output_file",
    "file",
    "path",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class RawPredictionRow:
    """One image-level row read from raw_predictions.jsonl."""

    line_index: int
    row: dict[str, Any]
    image_value: str | None
    image_path: Path | None
    width: int | None
    height: int | None
    predictions: list[dict[str, Any]]


@dataclass(slots=True)
class ReviewItem:
    """One pending binary review item."""

    review_id: str
    review_unit: str
    raw_row: RawPredictionRow
    predictions: list[dict[str, Any]]
    raw_prediction_ref: dict[str, Any]
    target_prediction_index: int | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_list(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        if not value.strip():
            return set()
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    return {item for item in items if item}


def raw_image_value(row: dict[str, Any]) -> str | None:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def resolve_image_path(
    image_value: str | None,
    *,
    raw_predictions_file: Path,
    image_search_roots: list[Path],
) -> Path | None:
    if image_value is None:
        return None

    raw = Path(image_value).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                raw_predictions_file.parent / raw,
                raw_predictions_file.parent / "images" / raw,
                raw_predictions_file.parent.parent / raw,
                Path(to_absolute_path(image_value)),
            ]
        )
        candidates.extend(root / raw for root in image_search_roots)
        candidates.extend(root / raw.name for root in image_search_roots)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def read_image_size(path: Path | None, row: dict[str, Any]) -> tuple[int | None, int | None]:
    width = row.get("width")
    height = row.get("height")
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except (TypeError, ValueError):
            pass
    if path is None or not path.is_file():
        return None, None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def normalize_predictions(row: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = row.get("predictions")
    if not isinstance(predictions, list):
        return []
    return [dict(item) for item in predictions if isinstance(item, dict)]


def prediction_passes_filters(
    prediction: dict[str, Any],
    *,
    min_score: float,
    allowed_tasks: set[str],
    allowed_labels: set[str],
) -> bool:
    try:
        score = float(prediction.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score < min_score:
        return False
    if allowed_tasks and str(prediction.get("task", "")) not in allowed_tasks:
        return False
    if allowed_labels and str(prediction.get("label", "")) not in allowed_labels:
        return False
    return True


def load_raw_rows(
    *,
    raw_predictions_file: Path,
    image_search_roots: list[Path],
    min_score: float,
    allowed_tasks: set[str],
    allowed_labels: set[str],
) -> list[RawPredictionRow]:
    rows: list[RawPredictionRow] = []
    for line_index, row in enumerate(read_jsonl(raw_predictions_file), start=1):
        image_value = raw_image_value(row)
        image_path = resolve_image_path(
            image_value,
            raw_predictions_file=raw_predictions_file,
            image_search_roots=image_search_roots,
        )
        width, height = read_image_size(image_path, row)
        predictions = [
            prediction
            for prediction in normalize_predictions(row)
            if prediction_passes_filters(
                prediction,
                min_score=min_score,
                allowed_tasks=allowed_tasks,
                allowed_labels=allowed_labels,
            )
        ]
        rows.append(
            RawPredictionRow(
                line_index=line_index,
                row=row,
                image_value=image_value,
                image_path=image_path,
                width=width,
                height=height,
                predictions=predictions,
            )
        )
    return rows


def build_review_items(
    *,
    raw_rows: list[RawPredictionRow],
    raw_predictions_file: Path,
    review_unit: str,
    max_items: int,
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    if review_unit == "image":
        for raw_row in raw_rows:
            if not raw_row.predictions:
                continue
            next_index = len(items) + 1
            items.append(
                ReviewItem(
                    review_id=f"review_{next_index:06d}",
                    review_unit=review_unit,
                    raw_row=raw_row,
                    predictions=raw_row.predictions,
                    raw_prediction_ref={
                        "raw_predictions_file": str(raw_predictions_file),
                        "line_index": raw_row.line_index,
                        "prediction_index": None,
                        "image": raw_row.image_value,
                    },
                )
            )
    elif review_unit == "annotation":
        for raw_row in raw_rows:
            original_predictions = normalize_predictions(raw_row.row)
            for prediction in raw_row.predictions:
                prediction_index = find_prediction_index(original_predictions, prediction)
                next_index = len(items) + 1
                items.append(
                    ReviewItem(
                        review_id=f"review_{next_index:06d}",
                        review_unit=review_unit,
                        raw_row=raw_row,
                        predictions=[prediction],
                        raw_prediction_ref={
                            "raw_predictions_file": str(raw_predictions_file),
                            "line_index": raw_row.line_index,
                            "prediction_index": prediction_index,
                            "image": raw_row.image_value,
                        },
                        target_prediction_index=prediction_index,
                    )
                )
    else:
        raise ValueError("review_unit must be image or annotation")

    if max_items > 0:
        return items[:max_items]
    return items


def canonical_prediction_key(prediction: dict[str, Any]) -> str:
    return json.dumps(prediction, sort_keys=True, ensure_ascii=True)


def find_prediction_index(predictions: list[dict[str, Any]], target: dict[str, Any]) -> int | None:
    target_key = canonical_prediction_key(target)
    for index, prediction in enumerate(predictions):
        if canonical_prediction_key(prediction) == target_key:
            return index
    return None


def safe_label(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bbox_from_prediction(prediction: dict[str, Any]) -> list[float] | None:
    box = prediction.get("bbox_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def prediction_text(prediction: dict[str, Any]) -> str:
    parts = [
        safe_label(prediction.get("task")),
        safe_label(prediction.get("label")),
    ]
    try:
        parts.append(f"{float(prediction.get('score', 0.0)):.2f}")
    except (TypeError, ValueError):
        parts.append("0.00")
    teacher = safe_label(prediction.get("teacher"))
    if teacher:
        parts.append(teacher)
    return " ".join(part for part in parts if part)


def color_for_prediction(prediction: dict[str, Any], *, faded: bool = False) -> tuple[int, int, int]:
    palette = [
        (255, 75, 80),
        (45, 170, 255),
        (70, 205, 110),
        (255, 178, 55),
        (185, 115, 245),
        (245, 95, 190),
    ]
    key = f"{prediction.get('task', '')}/{prediction.get('label', '')}/{prediction.get('teacher', '')}"
    color = palette[sum(ord(char) for char in key) % len(palette)]
    if faded:
        return tuple(int(channel * 0.35 + 255 * 0.65) for channel in color)
    return color


def draw_label(
    *,
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: Any,
) -> None:
    x, y = xy
    y = max(0.0, y)
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle([left, top, right + 4, bottom + 2], fill=(0, 0, 0))
    except Exception:  # noqa: BLE001
        pass
    draw.text((x + 2, y), text, fill=color, font=font)


def draw_overlay(
    *,
    image_path: Path,
    output_path: Path,
    all_predictions: list[dict[str, Any]],
    highlighted_prediction: dict[str, Any] | None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    highlighted_key = canonical_prediction_key(highlighted_prediction) if highlighted_prediction is not None else None
    for prediction in all_predictions:
        box = bbox_from_prediction(prediction)
        if box is None:
            continue
        is_highlighted = highlighted_key is None or canonical_prediction_key(prediction) == highlighted_key
        color = color_for_prediction(prediction, faded=not is_highlighted)
        width = 4 if highlighted_key is not None and is_highlighted else 2
        if highlighted_key is not None and not is_highlighted:
            width = 1
        draw.rectangle(box, outline=color, width=width)
        if is_highlighted or highlighted_key is None:
            draw_label(
                draw=draw,
                xy=(box[0], box[1] - 14),
                text=prediction_text(prediction),
                color=color,
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def queue_row_from_item(
    *,
    item: ReviewItem,
    overlay_image: str | None,
) -> dict[str, Any]:
    return {
        "review_id": item.review_id,
        "review_unit": item.review_unit,
        "image": item.raw_row.image_value,
        "width": item.raw_row.width,
        "height": item.raw_row.height,
        "overlay_image": overlay_image,
        "raw_prediction_ref": item.raw_prediction_ref,
        "predictions": item.predictions,
        "status": "pending",
    }


def build_manifest(
    *,
    cfg: DictConfig,
    raw_predictions_file: Path,
    output_dir: Path,
    items: list[ReviewItem],
    missing_images: int,
    missing_image_examples: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for item in items:
        for prediction in item.predictions:
            task_counts[str(prediction.get("task", ""))] += 1
            label_counts[str(prediction.get("label", ""))] += 1

    return {
        "created_at": now_iso(),
        "dry_run": dry_run,
        "raw_predictions_file": str(raw_predictions_file),
        "review_unit": str(cfg.review_unit),
        "total_queue_items": len(items),
        "task_counts": dict(task_counts),
        "label_counts": dict(label_counts),
        "missing_images": missing_images,
        "missing_image_examples": missing_image_examples,
        "output_dir": str(output_dir),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def run_build_review_queue(cfg: DictConfig) -> dict[str, Any]:
    raw_predictions_file = Path(to_absolute_path(str(cfg.raw_predictions_file))).resolve()
    if not raw_predictions_file.is_file():
        raise FileNotFoundError(f"raw_predictions_file not found: {raw_predictions_file}")

    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    review_assets_dir = output_dir / "review_assets"
    review_unit = str(cfg.review_unit)
    if review_unit not in {"image", "annotation"}:
        raise ValueError("review_unit must be image or annotation")

    image_search_roots = [
        Path(to_absolute_path(str(value))).resolve()
        for value in cfg.image_search_roots
    ]
    allowed_tasks = normalize_list(cfg.allowed_tasks)
    allowed_labels = normalize_list(cfg.allowed_labels)
    raw_rows = load_raw_rows(
        raw_predictions_file=raw_predictions_file,
        image_search_roots=image_search_roots,
        min_score=float(cfg.min_score),
        allowed_tasks=allowed_tasks,
        allowed_labels=allowed_labels,
    )
    items = build_review_items(
        raw_rows=raw_rows,
        raw_predictions_file=raw_predictions_file,
        review_unit=review_unit,
        max_items=int(cfg.max_items),
    )
    missing_rows_by_line = {
        item.raw_row.line_index: item.raw_row
        for item in items
        if item.raw_row.image_path is None
    }
    missing_images = len(missing_rows_by_line)
    missing_image_examples = [
        str(row.image_value)
        for row in missing_rows_by_line.values()
        if row.image_value is not None
    ][: int(cfg.max_missing_image_examples)]

    manifest = build_manifest(
        cfg=cfg,
        raw_predictions_file=raw_predictions_file,
        output_dir=output_dir,
        items=items,
        missing_images=missing_images,
        missing_image_examples=missing_image_examples,
        dry_run=bool(cfg.dry_run),
    )

    if bool(cfg.fail_fast) and missing_images > 0:
        raise FileNotFoundError(
            f"{missing_images} raw prediction rows refer to missing images. "
            "Set fail_fast=false to write queue rows without overlays."
        )

    if bool(cfg.dry_run):
        summary = {
            "status": "dry_run",
            "raw_predictions_file": str(raw_predictions_file),
            "output_dir": str(output_dir),
            "review_unit": review_unit,
            "raw_rows": len(raw_rows),
            "queue_items": len(items),
            "missing_images": missing_images,
            "task_counts": manifest["task_counts"],
            "label_counts": manifest["label_counts"],
            "preview_review_ids": [item.review_id for item in items[: int(cfg.max_preview_items)]],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    queue_rows: list[dict[str, Any]] = []
    overlay_count = 0
    for index, item in enumerate(items, start=1):
        overlay_relative: str | None = None
        if item.raw_row.image_path is not None:
            overlay_path = review_assets_dir / f"overlay_{index:06d}.jpg"
            all_predictions = item.raw_row.predictions
            highlighted_prediction = item.predictions[0] if item.review_unit == "annotation" else None
            draw_overlay(
                image_path=item.raw_row.image_path,
                output_path=overlay_path,
                all_predictions=all_predictions,
                highlighted_prediction=highlighted_prediction,
            )
            overlay_relative = overlay_path.relative_to(output_dir).as_posix()
            overlay_count += 1
        queue_rows.append(queue_row_from_item(item=item, overlay_image=overlay_relative))

    write_jsonl(output_dir / "review_queue.jsonl", queue_rows)
    manifest["overlay_count"] = overlay_count
    write_json(output_dir / "manifest.json", manifest)

    result = {
        "status": "ok",
        "output_dir": str(output_dir),
        "review_queue": str(output_dir / "review_queue.jsonl"),
        "manifest": str(output_dir / "manifest.json"),
        "queue_items": len(queue_rows),
        "overlay_count": overlay_count,
        "missing_images": missing_images,
        "task_counts": manifest["task_counts"],
        "label_counts": manifest["label_counts"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="build_review_queue",
)
def main(cfg: DictConfig) -> None:
    run_build_review_queue(cfg)


if __name__ == "__main__":
    main()
