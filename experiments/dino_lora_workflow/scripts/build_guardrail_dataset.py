"""Overview:
Merge trusted Grounding DINO annotation JSONL files into one guardrail dataset.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/build_guardrail_dataset.py input_annotation_files='[data/dino_workflow/court/manual100/annotations.jsonl]'
    .venv/bin/python experiments/dino_lora_workflow/scripts/build_guardrail_dataset.py dry_run=true input_annotation_files='[outputs/tmp/a.jsonl,outputs/tmp/b.jsonl]'

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/build_guardrail_dataset.yaml`.
    - Guardrail datasets are trusted data mixed into every round of LoRA training.
    - Court, role, and ball annotations may be mixed in the same output JSONL; task and label fields distinguish them.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


IMAGE_KEYS = ("image", "image_path")
ROW_LEVEL_EXCLUDE_FOR_ANNOTATION = {
    "image",
    "image_path",
    "absolute_image_path",
    "width",
    "height",
    "split",
    "annotations",
    "label_source",
    "weight",
    "source",
    "guardrail",
    "source_annotation_file",
    "original_image_path",
}


@dataclass(slots=True)
class GuardrailEntry:
    """One normalized image row to write into the merged guardrail dataset."""

    row: dict[str, Any]
    source_annotation_file: Path
    original_image_path: Path
    destination_relative_image: str
    duplicate_original: bool = False
    missing_image: bool = False


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


def resolve_paths(values: Any, *, field_name: str) -> list[Path]:
    paths = [Path(to_absolute_path(str(value))).resolve() for value in values]
    if not paths:
        raise ValueError(f"{field_name} must contain at least one path")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{field_name} contains missing files: {missing}")
    return paths


def normalize_search_roots(values: Any) -> list[Path]:
    return [Path(to_absolute_path(str(value))).resolve() for value in values]


def row_image_value(row: dict[str, Any], *, source_file: Path) -> str:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"Row in {source_file} does not include `image` or `image_path`")


def resolve_image_path(
    image_value: str,
    *,
    source_file: Path,
    image_search_roots: list[Path],
) -> Path:
    candidate = Path(image_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    candidates = [
        Path(to_absolute_path(image_value)).resolve(),
        (source_file.parent / candidate).resolve(),
        (source_file.parent / "images" / candidate.name).resolve(),
    ]
    candidates.extend((root / candidate).resolve() for root in image_search_roots)
    candidates.extend((root / candidate.name).resolve() for root in image_search_roots)

    for item in candidates:
        if item.exists():
            return item

    return candidate.resolve() if candidate.is_absolute() else (source_file.parent / candidate).resolve()


def read_image_size(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return int(width), int(height)


def normalize_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = row.get("annotations")
    if isinstance(annotations, list):
        return [dict(item) for item in annotations if isinstance(item, dict)]
    if isinstance(annotations, dict):
        return [dict(annotations)]

    annotation = {
        key: value
        for key, value in row.items()
        if key not in ROW_LEVEL_EXCLUDE_FOR_ANNOTATION
    }
    return [annotation] if annotation else []


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def destination_image_relative_path(image_path: Path) -> str:
    suffix = image_path.suffix or ".jpg"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in image_path.stem)
    unique = short_hash(str(image_path.resolve()))
    return (Path("images") / f"{safe_stem}_{unique}{suffix.lower()}").as_posix()


def assign_splits(
    entries: list[GuardrailEntry],
    *,
    preserve_split: bool,
    val_ratio: float,
    seed: int,
) -> None:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be in [0, 1]")

    unset_indices = [
        index
        for index, entry in enumerate(entries)
        if not (preserve_split and str(entry.row.get("split") or "").strip())
    ]
    rng = random.Random(seed)
    rng.shuffle(unset_indices)
    val_count = int(round(len(unset_indices) * val_ratio))
    val_indices = set(unset_indices[:val_count])
    for index in unset_indices:
        entries[index].row["split"] = "val" if index in val_indices else "train"


def count_tasks(row: dict[str, Any]) -> list[str]:
    annotation_tasks = [
        str(annotation["task"])
        for annotation in row.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("task") is not None
    ]
    if annotation_tasks:
        return annotation_tasks
    return [str(row["task"])] if row.get("task") is not None else []


def count_labels(row: dict[str, Any]) -> list[str]:
    annotation_labels = [
        str(annotation["label"])
        for annotation in row.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("label") is not None
    ]
    if annotation_labels:
        return annotation_labels
    return [str(row["label"])] if row.get("label") is not None else []


def normalize_entry(
    row: dict[str, Any],
    *,
    source_file: Path,
    image_search_roots: list[Path],
    seen_original_images: set[str],
    default_weight: float,
    default_label_source: str,
    preserve_label_source: bool,
) -> GuardrailEntry:
    image_value = row_image_value(row, source_file=source_file)
    original_image_path = resolve_image_path(
        image_value,
        source_file=source_file,
        image_search_roots=image_search_roots,
    )
    original_key = str(original_image_path)
    duplicate_original = original_key in seen_original_images
    seen_original_images.add(original_key)

    width = row.get("width")
    height = row.get("height")
    if width is None or height is None:
        read_width, read_height = read_image_size(original_image_path)
        width = width if width is not None else read_width
        height = height if height is not None else read_height

    normalized = {
        key: value
        for key, value in row.items()
        if key not in {"image", "image_path", "absolute_image_path"}
    }
    normalized["image"] = destination_image_relative_path(original_image_path)
    if width is not None:
        normalized["width"] = int(width)
    if height is not None:
        normalized["height"] = int(height)
    if not (preserve_label_source and row.get("label_source") is not None):
        normalized["label_source"] = default_label_source
    normalized["guardrail"] = True
    normalized["weight"] = float(row.get("weight", default_weight))
    normalized["source_annotation_file"] = str(source_file)
    normalized["original_image_path"] = str(original_image_path)
    normalized["annotations"] = normalize_annotations(row)

    return GuardrailEntry(
        row=normalized,
        source_annotation_file=source_file,
        original_image_path=original_image_path,
        destination_relative_image=str(normalized["image"]),
        duplicate_original=duplicate_original,
        missing_image=not original_image_path.is_file(),
    )


def collect_entries(
    *,
    input_annotation_files: list[Path],
    image_search_roots: list[Path],
    default_weight: float,
    default_label_source: str,
    preserve_label_source: bool,
    fail_fast: bool,
) -> tuple[list[GuardrailEntry], list[dict[str, Any]], int]:
    entries: list[GuardrailEntry] = []
    missing_images: list[dict[str, Any]] = []
    seen_original_images: set[str] = set()
    total_rows = 0

    for source_file in input_annotation_files:
        rows = read_jsonl(source_file)
        total_rows += len(rows)
        for row_index, row in enumerate(rows, start=1):
            entry = normalize_entry(
                row,
                source_file=source_file,
                image_search_roots=image_search_roots,
                seen_original_images=seen_original_images,
                default_weight=default_weight,
                default_label_source=default_label_source,
                preserve_label_source=preserve_label_source,
            )
            if entry.missing_image:
                missing = {
                    "source_annotation_file": str(source_file),
                    "row_index": row_index,
                    "image": row_image_value(row, source_file=source_file),
                    "resolved_image_path": str(entry.original_image_path),
                }
                missing_images.append(missing)
                if fail_fast:
                    raise FileNotFoundError(f"Missing image for guardrail row: {missing}")
                continue
            entries.append(entry)
    return entries, missing_images, total_rows


def copy_images(
    *,
    entries: list[GuardrailEntry],
    output_dir: Path,
    duplicate_policy: str,
) -> tuple[int, int]:
    if duplicate_policy not in {"skip", "existing_overwrite", "error"}:
        raise ValueError("duplicate_policy must be one of: skip, existing_overwrite, error")

    copied = 0
    duplicate_count = 0
    copied_destinations: set[str] = set()
    for entry in entries:
        destination = output_dir / entry.destination_relative_image
        destination_key = str(destination)
        is_duplicate = entry.duplicate_original or destination_key in copied_destinations or destination.exists()
        if is_duplicate:
            duplicate_count += 1
            if duplicate_policy == "error":
                raise FileExistsError(f"Duplicate guardrail image destination: {destination}")
            if duplicate_policy == "skip":
                copied_destinations.add(destination_key)
                continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.original_image_path, destination)
        copied += 1
        copied_destinations.add(destination_key)
    return copied, duplicate_count


def build_manifest(
    *,
    cfg: DictConfig,
    output_dir: Path,
    input_annotation_files: list[Path],
    entries: list[GuardrailEntry],
    total_rows: int,
    copied_images: int,
    missing_images: list[dict[str, Any]],
    duplicate_count: int,
) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for entry in entries:
        split_counts[str(entry.row.get("split", "unknown"))] += 1
        task_counts.update(count_tasks(entry.row))
        label_counts.update(count_labels(entry.row))

    return {
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "input_annotation_files": [str(path) for path in input_annotation_files],
        "total_rows": total_rows,
        "written_rows": len(entries),
        "copied_images": copied_images,
        "missing_images": missing_images,
        "missing_image_count": len(missing_images),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "duplicate_count": duplicate_count,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def summarize_manifest(manifest: dict[str, Any], *, dry_run: bool, max_preview_rows: int) -> dict[str, Any]:
    summary = {
        "dry_run": dry_run,
        "output_dir": manifest["output_dir"],
        "input_annotation_files": manifest["input_annotation_files"],
        "total_rows": manifest["total_rows"],
        "written_rows": manifest["written_rows"],
        "copied_images": manifest["copied_images"],
        "missing_image_count": manifest["missing_image_count"],
        "split_counts": manifest["split_counts"],
        "task_counts": manifest["task_counts"],
        "label_counts": manifest["label_counts"],
        "duplicate_count": manifest["duplicate_count"],
    }
    if manifest["missing_images"]:
        summary["missing_images_preview"] = manifest["missing_images"][:max_preview_rows]
    return summary


def build_guardrail_dataset(cfg: DictConfig) -> dict[str, Any]:
    input_annotation_files = resolve_paths(
        cfg.input_annotation_files,
        field_name="input_annotation_files",
    )
    image_search_roots = normalize_search_roots(cfg.image_search_roots)
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()

    entries, missing_images, total_rows = collect_entries(
        input_annotation_files=input_annotation_files,
        image_search_roots=image_search_roots,
        default_weight=float(cfg.default_weight),
        default_label_source=str(cfg.default_label_source),
        preserve_label_source=bool(cfg.preserve_label_source),
        fail_fast=bool(cfg.fail_fast),
    )
    assign_splits(
        entries,
        preserve_split=bool(cfg.preserve_split),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
    )

    if bool(cfg.dry_run):
        manifest = build_manifest(
            cfg=cfg,
            output_dir=output_dir,
            input_annotation_files=input_annotation_files,
            entries=entries,
            total_rows=total_rows,
            copied_images=0,
            missing_images=missing_images,
            duplicate_count=sum(1 for entry in entries if entry.duplicate_original),
        )
        return summarize_manifest(
            manifest,
            dry_run=True,
            max_preview_rows=int(cfg.max_preview_rows),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_images, duplicate_count = copy_images(
        entries=entries,
        output_dir=output_dir,
        duplicate_policy=str(cfg.duplicate_policy),
    )
    write_jsonl(output_dir / "annotations.jsonl", [entry.row for entry in entries])
    manifest = build_manifest(
        cfg=cfg,
        output_dir=output_dir,
        input_annotation_files=input_annotation_files,
        entries=entries,
        total_rows=total_rows,
        copied_images=copied_images,
        missing_images=missing_images,
        duplicate_count=duplicate_count,
    )
    write_json(output_dir / "manifest.json", manifest)
    return summarize_manifest(
        manifest,
        dry_run=False,
        max_preview_rows=int(cfg.max_preview_rows),
    )


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="build_guardrail_dataset",
)
def main(cfg: DictConfig) -> None:
    summary = build_guardrail_dataset(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
