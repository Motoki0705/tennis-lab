"""Overview:
Convert point annotations into Grounding DINO bbox JSONL rows.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py source_format=tennis_center_csv input_paths='[data/tennis]' output_dir=data/dino_workflow/tennis/ball_guardrail
    .venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py source_format=court_kp_json input_paths='[data/court/data_train.json]' images_dir=data/court/images output_dir=data/dino_workflow/court/kp14_guardrail
    .venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py source_format=court_kp20_points_jsonl input_paths='[outputs/dino_workflow/court/manual100_kp20/annotations_points.jsonl]' output_dir=data/dino_workflow/court/manual100_kp20

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/convert_points_to_gdino_boxes.yaml`.
    - Supported source formats are `tennis_center_csv`, `court_kp_json`, and `court_kp20_points_jsonl`.
    - Each point becomes a fixed-size square bbox clipped to the source image bounds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "utils" / "schema" / "court.py").is_file():
            return candidate
    return start.parents[3]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.schema.court import COURT_KP_NAMES, NUM_COURT_KP  # noqa: E402


IMAGE_EXTENSIONS = ("", ".png", ".jpg", ".jpeg", ".webp", ".bmp")
SUPPORTED_SOURCE_FORMATS = {
    "tennis_center_csv",
    "court_kp_json",
    "court_kp20_points_jsonl",
}


@dataclass(slots=True)
class PointRecord:
    image_path: Path
    width: int | None
    height: int | None
    task: str
    source: dict[str, Any]
    points: list[dict[str, Any]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_plain_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def path_for_output(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(get_original_cwd()).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_config_paths(values: Any, *, field_name: str) -> list[Path]:
    paths = [Path(to_absolute_path(str(value))).resolve() for value in values]
    if not paths:
        raise ValueError(f"{field_name} must contain at least one path")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{field_name} contains missing paths: {missing}")
    return paths


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return Path(to_absolute_path(text)).resolve()


def read_image_size(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return int(width), int(height)


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if result == result else None


def visibility_value(value: Any) -> float:
    parsed = numeric(value)
    return float(parsed) if parsed is not None else 0.0


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def destination_image_relative_path(image_path: Path) -> str:
    suffix = image_path.suffix.lower() or ".jpg"
    unique = short_hash(str(image_path.resolve()))
    return (Path("images") / f"{safe_stem(image_path)}_{unique}{suffix}").as_posix()


def copy_image_if_needed(
    source: Path,
    *,
    output_dir: Path,
    copy_images: bool,
) -> str:
    if not copy_images:
        return path_for_output(source)
    relative_path = destination_image_relative_path(source)
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return relative_path


def expand_tennis_label_csv_paths(input_paths: list[Path]) -> list[Path]:
    label_paths: list[Path] = []
    for path in input_paths:
        if path.is_file():
            if path.name != "Label.csv":
                raise ValueError(f"tennis_center_csv file input must be Label.csv: {path}")
            label_paths.append(path)
        elif path.is_dir():
            direct = path / "Label.csv"
            if direct.is_file():
                label_paths.append(direct)
            label_paths.extend(sorted(path.glob("**/Label.csv")))
        else:
            raise FileNotFoundError(path)
    return sorted(set(label_paths))


def expand_json_files(input_paths: list[Path], *, suffix: str | None = None) -> list[Path]:
    files: list[Path] = []
    for path in input_paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            pattern = f"**/*{suffix}" if suffix else "**/*"
            files.extend(sorted(item for item in path.glob(pattern) if item.is_file()))
        else:
            raise FileNotFoundError(path)
    return sorted(set(files))


def resolve_image_with_extensions(images_dir: Path, image_id: str) -> Path | None:
    raw = Path(image_id)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    if raw.suffix:
        candidates.append(images_dir / raw.name)
        candidates.append(images_dir / raw)
    else:
        for extension in IMAGE_EXTENSIONS:
            if extension:
                candidates.append(images_dir / f"{image_id}{extension}")
            else:
                candidates.append(images_dir / image_id)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def infer_court_images_dir(input_file: Path, cfg_images_dir: Path | None) -> Path:
    if cfg_images_dir is not None:
        return cfg_images_dir
    candidates = [
        input_file.parent / "images",
        input_file.parent.parent / "images",
        Path(to_absolute_path("data/court/images")).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def normalize_point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = numeric(value[0])
    y = numeric(value[1])
    if x is None or y is None:
        return None
    return [float(x), float(y)]


def court_query(label: str) -> str:
    return f"{label.replace('_', ' ')} court keypoint"


def tennis_point_to_record(
    label_csv: Path,
    *,
    cfg: DictConfig,
) -> list[PointRecord]:
    records: list[PointRecord] = []
    with label_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"file name", "visibility", "x-coordinate", "y-coordinate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{label_csv} is missing required columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=2):
            if visibility_value(row.get("visibility")) <= 0:
                continue
            x = numeric(row.get("x-coordinate"))
            y = numeric(row.get("y-coordinate"))
            if x is None or y is None:
                continue
            frame_name = str(row.get("file name") or "").strip()
            if not frame_name:
                continue
            image_path = (label_csv.parent / frame_name).resolve()
            width, height = read_image_size(image_path)
            records.append(
                PointRecord(
                    image_path=image_path,
                    width=width,
                    height=height,
                    task="tennis_ball",
                    source={
                        "source_format": "tennis_center_csv",
                        "label_csv": path_for_output(label_csv),
                        "row_number": row_index,
                        "file_name": frame_name,
                        "visibility": row.get("visibility"),
                        "status": row.get("status"),
                    },
                    points=[
                        {
                            "task": "tennis_ball",
                            "label": "tennis_ball",
                            "query": "tennis ball",
                            "point_xy": [float(x), float(y)],
                        }
                    ],
                )
            )
            if int(cfg.max_records) > 0 and len(records) >= int(cfg.max_records):
                return records
    return records


def collect_tennis_records(input_paths: list[Path], *, cfg: DictConfig) -> list[PointRecord]:
    records: list[PointRecord] = []
    for label_csv in expand_tennis_label_csv_paths(input_paths):
        records.extend(tennis_point_to_record(label_csv, cfg=cfg))
        if int(cfg.max_records) > 0 and len(records) >= int(cfg.max_records):
            return records[: int(cfg.max_records)]
    return records


def court_json_to_records(
    input_file: Path,
    *,
    cfg: DictConfig,
    images_dir: Path | None,
) -> list[PointRecord]:
    records: list[PointRecord] = []
    data = load_json(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of court rows in {input_file}")
    resolved_images_dir = infer_court_images_dir(input_file, images_dir)
    for row_index, item in enumerate(data, start=1):
        if not isinstance(item, dict) or "id" not in item:
            continue
        kps = item.get("kps")
        if not isinstance(kps, list):
            continue
        image_id = str(item["id"])
        image_path = resolve_image_with_extensions(resolved_images_dir, image_id)
        placeholder = resolved_images_dir / f"{image_id}.png"
        width, height = read_image_size(image_path) if image_path else (None, None)
        points: list[dict[str, Any]] = []
        for index, point_value in enumerate(kps[:14]):
            point = normalize_point(point_value)
            if point is None:
                continue
            label = COURT_KP_NAMES[index]
            points.append(
                {
                    "task": "court_kp14",
                    "label": label,
                    "query": court_query(label),
                    "point_xy": point,
                    "kp_index": index,
                }
            )
        if not points:
            continue
        records.append(
            PointRecord(
                image_path=(image_path or placeholder).resolve(),
                width=width,
                height=height,
                task="court_kp14",
                source={
                    "source_format": "court_kp_json",
                    "input_json": path_for_output(input_file),
                    "row_number": row_index,
                    "id": image_id,
                    "metric": item.get("metric"),
                    "images_dir": path_for_output(resolved_images_dir),
                },
                points=points,
            )
        )
        if int(cfg.max_records) > 0 and len(records) >= int(cfg.max_records):
            return records
    return records


def collect_court_json_records(
    input_paths: list[Path],
    *,
    cfg: DictConfig,
    images_dir: Path | None,
) -> list[PointRecord]:
    records: list[PointRecord] = []
    for input_file in expand_json_files(input_paths, suffix=".json"):
        records.extend(court_json_to_records(input_file, cfg=cfg, images_dir=images_dir))
        if int(cfg.max_records) > 0 and len(records) >= int(cfg.max_records):
            return records[: int(cfg.max_records)]
    return records


def resolve_row_image_path(row: dict[str, Any], source_file: Path) -> Path:
    value = row.get("image_path") or row.get("image") or row.get("absolute_image_path")
    if value is None:
        raise ValueError(f"Manual KP20 row in {source_file} has no image path")
    raw = Path(str(value))
    if raw.is_absolute():
        return raw.resolve()
    candidates = [
        Path(to_absolute_path(str(value))).resolve(),
        (source_file.parent / raw).resolve(),
        (source_file.parent / "images" / raw.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def manual_kp20_jsonl_to_records(input_file: Path, *, cfg: DictConfig) -> list[PointRecord]:
    records: list[PointRecord] = []
    for row_index, row in enumerate(load_jsonl(input_file), start=1):
        kps20 = row.get("kps20")
        if not isinstance(kps20, list):
            continue
        image_path = resolve_row_image_path(row, input_file)
        width = int(row["width"]) if row.get("width") is not None else None
        height = int(row["height"]) if row.get("height") is not None else None
        disk_width, disk_height = read_image_size(image_path)
        width = disk_width or width
        height = disk_height or height
        points: list[dict[str, Any]] = []
        for index, point_value in enumerate(kps20[:NUM_COURT_KP]):
            point = normalize_point(point_value)
            if point is None:
                continue
            label = COURT_KP_NAMES[index]
            points.append(
                {
                    "task": "court_kp20",
                    "label": label,
                    "query": court_query(label),
                    "point_xy": point,
                    "kp_index": index,
                }
            )
        if not points:
            continue
        records.append(
            PointRecord(
                image_path=image_path,
                width=width,
                height=height,
                task="court_kp20",
                source={
                    "source_format": "court_kp20_points_jsonl",
                    "input_jsonl": path_for_output(input_file),
                    "row_number": row_index,
                    "id": row.get("id"),
                    "schema_name": row.get("schema_name"),
                    "source": row.get("source"),
                },
                points=points,
            )
        )
        if int(cfg.max_records) > 0 and len(records) >= int(cfg.max_records):
            return records
    return records


def collect_manual_kp20_records(input_paths: list[Path], *, cfg: DictConfig) -> list[PointRecord]:
    records: list[PointRecord] = []
    for input_file in expand_json_files(input_paths, suffix=".jsonl"):
        records.extend(manual_kp20_jsonl_to_records(input_file, cfg=cfg))
        if int(cfg.max_records) > 0 and len(records) >= int(cfg.max_records):
            return records[: int(cfg.max_records)]
    return records


def box_size_for(point: dict[str, Any], cfg: DictConfig) -> float:
    label = str(point["label"])
    task = str(point["task"])
    overrides = cfg.get("box_size_overrides") or {}
    per_label = overrides.get("per_label") or {}
    per_task = overrides.get("per_task") or {}
    if label in per_label:
        return float(per_label[label])
    if task in per_task:
        return float(per_task[task])
    return float(cfg.box_size_px)


def point_to_bbox(
    point_xy: list[float],
    *,
    width: int,
    height: int,
    box_size: float,
    min_box_size: float,
) -> list[float] | None:
    x, y = float(point_xy[0]), float(point_xy[1])
    if not (x == x and y == y and box_size > 0):
        return None
    half = box_size / 2.0
    x1 = max(0.0, min(float(width), x - half))
    y1 = max(0.0, min(float(height), y - half))
    x2 = max(0.0, min(float(width), x + half))
    y2 = max(0.0, min(float(height), y + half))
    if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def row_split(index: int, cfg: DictConfig, val_indices: set[int]) -> str:
    configured = cfg.get("split")
    if configured is not None and str(configured).strip():
        return str(configured)
    return "val" if index in val_indices else "train"


def make_val_indices(count: int, cfg: DictConfig) -> set[int]:
    val_ratio = float(cfg.val_ratio)
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be in [0, 1]")
    if cfg.get("split") is not None and str(cfg.split).strip():
        return set()
    indices = list(range(count))
    rng = random.Random(int(cfg.seed))
    rng.shuffle(indices)
    val_count = int(round(count * val_ratio))
    return set(indices[:val_count])


def records_to_rows(
    records: list[PointRecord],
    *,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing_images: list[dict[str, Any]] = []
    dropped_annotations: list[dict[str, Any]] = []
    val_indices = make_val_indices(len(records), cfg)
    min_box_size = float(cfg.min_box_size_px)

    for index, record in enumerate(records):
        if not record.image_path.is_file():
            missing_images.append({"image_path": path_for_output(record.image_path), "source": record.source})
            continue
        width = record.width
        height = record.height
        if width is None or height is None:
            width, height = read_image_size(record.image_path)
        if width is None or height is None:
            missing_images.append({"image_path": path_for_output(record.image_path), "source": record.source})
            continue

        annotations: list[dict[str, Any]] = []
        for point_index, point in enumerate(record.points):
            bbox = point_to_bbox(
                point["point_xy"],
                width=width,
                height=height,
                box_size=box_size_for(point, cfg),
                min_box_size=min_box_size,
            )
            if bbox is None:
                dropped_annotations.append(
                    {
                        "reason": "bbox_too_small_or_invalid",
                        "image_path": path_for_output(record.image_path),
                        "point_index": point_index,
                        "point": point,
                    }
                )
                continue
            annotation = {
                "task": point["task"],
                "label": point["label"],
                "query": point["query"],
                "bbox_xyxy": bbox,
                "point_xy": [round(float(point["point_xy"][0]), 4), round(float(point["point_xy"][1]), 4)],
            }
            if point.get("kp_index") is not None:
                annotation["kp_index"] = int(point["kp_index"])
            if point.get("score") is not None:
                annotation["score"] = float(point["score"])
            annotations.append(annotation)

        if not annotations:
            continue

        image_value = copy_image_if_needed(
            record.image_path,
            output_dir=output_dir,
            copy_images=bool(cfg.copy_images) and not bool(cfg.dry_run),
        )
        rows.append(
            {
                "image": image_value,
                "width": int(width),
                "height": int(height),
                "task": record.task,
                "label_source": str(cfg.label_source),
                "weight": float(cfg.weight),
                "split": row_split(index, cfg, val_indices),
                "annotations": annotations,
                "source": {
                    **record.source,
                    "original_image_path": path_for_output(record.image_path),
                },
            }
        )

    return rows, missing_images, dropped_annotations


def collect_records(cfg: DictConfig) -> list[PointRecord]:
    source_format = str(cfg.source_format)
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(f"Unsupported source_format={source_format!r}; expected one of {sorted(SUPPORTED_SOURCE_FORMATS)}")
    input_paths = resolve_config_paths(cfg.input_paths, field_name="input_paths")
    images_dir = optional_path(cfg.get("images_dir"))
    if source_format == "tennis_center_csv":
        return collect_tennis_records(input_paths, cfg=cfg)
    if source_format == "court_kp_json":
        return collect_court_json_records(input_paths, cfg=cfg, images_dir=images_dir)
    if source_format == "court_kp20_points_jsonl":
        return collect_manual_kp20_records(input_paths, cfg=cfg)
    raise AssertionError(source_format)


def build_manifest(
    *,
    cfg: DictConfig,
    input_paths: list[Path],
    rows: list[dict[str, Any]],
    missing_images: list[dict[str, Any]],
    dropped_annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        for annotation in row.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            task_counts[str(annotation.get("task"))] += 1
            label_counts[str(annotation.get("label"))] += 1

    return {
        "created_at": now_iso(),
        "source_format": str(cfg.source_format),
        "input_paths": [path_for_output(path) for path in input_paths],
        "output_dir": str(cfg.output_dir),
        "output_annotations_file": str(cfg.output_annotations_file),
        "total_images": len(rows),
        "total_annotations": sum(len(row.get("annotations", [])) for row in rows),
        "task_counts": dict(sorted(task_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "missing_images": missing_images,
        "missing_images_count": len(missing_images),
        "dropped_annotations": dropped_annotations,
        "dropped_annotations_count": len(dropped_annotations),
        "box_size_px": as_plain_container(cfg).get("box_size_px"),
        "box_size_overrides": as_plain_container(cfg).get("box_size_overrides"),
        "config": as_plain_container(cfg),
    }


def summarize_manifest(manifest: dict[str, Any], *, dry_run: bool, max_preview_rows: int) -> dict[str, Any]:
    summary = {
        "dry_run": dry_run,
        "source_format": manifest["source_format"],
        "total_images": manifest["total_images"],
        "total_annotations": manifest["total_annotations"],
        "task_counts": manifest["task_counts"],
        "label_counts": manifest["label_counts"],
        "missing_images_count": manifest["missing_images_count"],
        "dropped_annotations_count": manifest["dropped_annotations_count"],
        "output_dir": manifest["output_dir"],
        "output_annotations_file": manifest["output_annotations_file"],
    }
    if max_preview_rows > 0:
        summary["missing_images_preview"] = manifest["missing_images"][:max_preview_rows]
        summary["dropped_annotations_preview"] = manifest["dropped_annotations"][:max_preview_rows]
    return summary


def convert_points_to_gdino_boxes(cfg: DictConfig) -> dict[str, Any]:
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    input_paths = resolve_config_paths(cfg.input_paths, field_name="input_paths")
    records = collect_records(cfg)
    rows, missing_images, dropped_annotations = records_to_rows(
        records,
        cfg=cfg,
        output_dir=output_dir,
    )
    manifest = build_manifest(
        cfg=cfg,
        input_paths=input_paths,
        rows=rows,
        missing_images=missing_images,
        dropped_annotations=dropped_annotations,
    )

    if bool(cfg.dry_run):
        return summarize_manifest(manifest, dry_run=True, max_preview_rows=int(cfg.max_preview_rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / str(cfg.output_annotations_file)
    write_jsonl(annotations_path, rows)
    write_json(output_dir / "manifest.json", manifest)
    return summarize_manifest(manifest, dry_run=False, max_preview_rows=int(cfg.max_preview_rows))


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="convert_points_to_gdino_boxes",
)
def main(cfg: DictConfig) -> None:
    summary = convert_points_to_gdino_boxes(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
