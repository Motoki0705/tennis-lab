"""Overview:
Run one or more Grounding DINO LoRA teachers over an image set and write raw pseudo-label predictions.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/run_gdino_inference.py image_dir=data/dino_workflow/sources/youtube/frames teacher_manifest=experiments/dino_lora_workflow/configs/teacher_manifest.example.json
    .venv/bin/python experiments/dino_lora_workflow/scripts/run_gdino_inference.py dry_run=true image_manifest=data/dino_workflow/sources/youtube/frames/manifest.json

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/run_gdino_inference.yaml`.
    - The teacher manifest controls court-only, roles-only, or mixed teacher inference without domain-specific scripts.
    - Adapter archives are `.tar.zst` files unpacked with the workflow archive unpacker before model loading.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Workaround for PEFT ≥0.18: GroundingDinoModel lacks get_input_embeddings
import transformers as _transformers
import torch.nn as _nn


class __DummyEmbedding(_nn.Module):
    def parameters(self, recurse=True):
        return iter([])


def __get_input_embeddings(self):
    return __DummyEmbedding()


_transformers.GroundingDinoModel.get_input_embeddings = __get_input_embeddings


LOGGER = logging.getLogger("dino_lora_inference")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class ImageItem:
    """One resolved image to infer."""

    path: Path
    image_value: str
    manifest_row: dict[str, Any] | None


@dataclass(slots=True)
class TeacherSpec:
    """One teacher entry from the teacher manifest."""

    name: str
    task: str
    base_model: str
    adapter_dir: Path | None
    adapter_archive: Path | None
    processor_dir: Path | None
    queries_file: Path
    queries: list[str]
    threshold: float
    text_threshold: float
    nms_threshold: float
    nms_mode: str
    max_detections_per_image: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(to_absolute_path(text)).expanduser().resolve()


def resolve_path(value: Any, *, base_dirs: list[Path], required: bool, field_name: str) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        if required:
            raise ValueError(f"{field_name} is required")
        return None

    raw = Path(text).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    for base_dir in base_dirs:
        candidates.append(base_dir / raw)
    candidates.append(Path(to_absolute_path(text)))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    if required:
        raise FileNotFoundError(f"{field_name} not found: {text}")
    return Path(to_absolute_path(text)).resolve()


def read_queries(path: Path) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            query = stripped.strip(".").strip()
            key = normalize_query(query)
            if not key or key in seen:
                continue
            seen.add(key)
            queries.append(query)
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def normalize_query(value: str) -> str:
    return value.strip().strip(".").strip().lower()


def normalize_label(value: str) -> str:
    text = normalize_query(value)
    for suffix in (" court keypoint", " keypoint"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text or "unknown"


def prompt_from_queries(queries: list[str]) -> str:
    return " ".join(f"{query.strip().strip('.')}." for query in queries)


def image_candidate_from_row(row: dict[str, Any]) -> str | None:
    for key in (
        "absolute_output_file",
        "absolute_image_path",
        "image_path",
        "image",
        "output_file",
        "file",
        "path",
        "relative_path",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def resolve_image_from_row(row: dict[str, Any], *, manifest_path: Path, manifest_payload: dict[str, Any] | None) -> Path | None:
    value = image_candidate_from_row(row)
    if value is None:
        return None
    raw = Path(value).expanduser()
    base_dirs = [manifest_path.parent]
    if manifest_payload is not None and manifest_payload.get("output_dir"):
        base_dirs.insert(0, Path(str(manifest_payload["output_dir"])).expanduser())
    if manifest_path.parent.parent not in base_dirs:
        base_dirs.append(manifest_path.parent.parent)
    candidates: list[Path] = [raw] if raw.is_absolute() else []
    candidates.extend(base_dir / raw for base_dir in base_dirs)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve() if candidates else None


def rows_from_json_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
    for key in ("frames", "images", "entries", "files"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return payload, [row for row in rows if isinstance(row, dict)]
    if image_candidate_from_row(payload) is not None:
        return payload, [payload]
    raise ValueError(f"No frames/images/entries/files list found in image manifest: {path}")


def collect_images_from_manifest(path: Path) -> list[ImageItem]:
    if path.suffix.lower() == ".jsonl":
        manifest_payload: dict[str, Any] | None = None
        rows = read_jsonl(path)
    else:
        manifest_payload, rows = rows_from_json_manifest(path)

    items: list[ImageItem] = []
    seen: set[Path] = set()
    for row in rows:
        resolved = resolve_image_from_row(row, manifest_path=path, manifest_payload=manifest_payload)
        if resolved is None or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not resolved.is_file():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        image_value = image_candidate_from_row(row) or str(resolved)
        items.append(ImageItem(path=resolved, image_value=str(image_value), manifest_row=row))
    if not items:
        raise FileNotFoundError(f"No images resolved from image manifest: {path}")
    return sorted(items, key=lambda item: str(item.path))


def collect_images_from_dir(path: Path) -> list[ImageItem]:
    if not path.is_dir():
        raise FileNotFoundError(f"image_dir is not a directory: {path}")
    items = [
        ImageItem(path=image_path.resolve(), image_value=str(image_path.relative_to(path)), manifest_row=None)
        for image_path in sorted(path.rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not items:
        raise FileNotFoundError(f"No images found under image_dir: {path}")
    return items


def collect_images(cfg: DictConfig) -> tuple[list[ImageItem], dict[str, Any]]:
    image_manifest = optional_path(cfg.image_manifest)
    image_dir = optional_path(cfg.image_dir)
    if image_manifest is not None:
        images = collect_images_from_manifest(image_manifest)
        source = {"type": "manifest", "path": str(image_manifest)}
    elif image_dir is not None:
        images = collect_images_from_dir(image_dir)
        source = {"type": "directory", "path": str(image_dir)}
    else:
        raise ValueError("Set either image_manifest or image_dir")

    max_images = int(cfg.max_images)
    if max_images > 0:
        images = images[:max_images]
    return images, source


def parse_teacher_manifest(cfg: DictConfig) -> tuple[Path, dict[str, Any], list[TeacherSpec]]:
    manifest_path = optional_path(cfg.teacher_manifest)
    if manifest_path is None:
        raise ValueError("teacher_manifest is required")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"teacher_manifest not found: {manifest_path}")

    payload = read_json(manifest_path)
    teachers_payload = payload.get("teachers")
    if not isinstance(teachers_payload, list) or not teachers_payload:
        raise ValueError(f"teacher_manifest must include a non-empty teachers list: {manifest_path}")

    teachers: list[TeacherSpec] = []
    base_dirs = [manifest_path.parent, Path.cwd()]
    for index, teacher in enumerate(teachers_payload, start=1):
        if not isinstance(teacher, dict):
            raise ValueError(f"teachers[{index}] must be an object")
        name = str(teacher.get("name") or f"teacher_{index:02d}")
        task = str(teacher.get("task") or "").strip()
        base_model = str(teacher.get("base_model") or "").strip()
        if not task:
            raise ValueError(f"Teacher {name} is missing task")
        if not base_model:
            raise ValueError(f"Teacher {name} is missing base_model")

        queries_file = resolve_path(
            teacher.get("queries_file"),
            base_dirs=base_dirs,
            required=True,
            field_name=f"teachers[{index}].queries_file",
        )
        assert queries_file is not None
        adapter_dir = resolve_path(
            teacher.get("adapter_dir"),
            base_dirs=base_dirs,
            required=False,
            field_name=f"teachers[{index}].adapter_dir",
        )
        adapter_archive = resolve_path(
            teacher.get("adapter_archive"),
            base_dirs=base_dirs,
            required=False,
            field_name=f"teachers[{index}].adapter_archive",
        )
        processor_dir = resolve_path(
            teacher.get("processor_dir"),
            base_dirs=base_dirs,
            required=False,
            field_name=f"teachers[{index}].processor_dir",
        )
        if adapter_dir is not None and adapter_archive is not None:
            raise ValueError(f"Teacher {name} sets both adapter_dir and adapter_archive")

        teachers.append(
            TeacherSpec(
                name=name,
                task=task,
                base_model=base_model,
                adapter_dir=adapter_dir,
                adapter_archive=adapter_archive,
                processor_dir=processor_dir,
                queries_file=queries_file,
                queries=read_queries(queries_file),
                threshold=float(teacher.get("threshold", cfg.threshold)),
                text_threshold=float(teacher.get("text_threshold", cfg.text_threshold)),
                nms_threshold=float(teacher.get("nms_threshold", cfg.nms_threshold)),
                nms_mode=str(teacher.get("nms_mode", cfg.nms_mode)).lower(),
                max_detections_per_image=int(
                    teacher.get("max_detections_per_image", cfg.max_detections_per_image)
                ),
            )
        )
    return manifest_path, payload, teachers


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def clip_box_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return None
    return [x1, y1, x2, y2]


def box_iou(box: list[float], boxes: list[list[float]]) -> list[float]:
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ious: list[float] = []
    for other in boxes:
        ox1, oy1, ox2, oy2 = other
        inter_x1 = max(x1, ox1)
        inter_y1 = max(y1, oy1)
        inter_x2 = min(x2, ox2)
        inter_y2 = min(y2, oy2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        union = area + other_area - inter_area
        ious.append(0.0 if union <= 0.0 else inter_area / union)
    return ious


def nms_keep_indices(predictions: list[dict[str, Any]], threshold: float) -> list[int]:
    if threshold <= 0.0 or len(predictions) <= 1:
        return list(range(len(predictions)))
    order = sorted(range(len(predictions)), key=lambda index: float(predictions[index]["score"]), reverse=True)
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining_boxes = [predictions[index]["bbox_xyxy"] for index in order]
        ious = box_iou(predictions[current]["bbox_xyxy"], remaining_boxes)
        order = [index for index, iou in zip(order, ious, strict=True) if iou <= threshold]
    return keep


def apply_nms(predictions: list[dict[str, Any]], *, threshold: float, mode: str, max_detections: int) -> list[dict[str, Any]]:
    if not predictions:
        return []
    if mode not in {"per_label", "all"}:
        raise ValueError("nms_mode must be per_label or all")

    kept: list[dict[str, Any]] = []
    if mode == "all":
        kept = [predictions[index] for index in nms_keep_indices(predictions, threshold)]
    else:
        by_label: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, prediction in enumerate(predictions):
            by_label.setdefault(str(prediction["label"]), []).append((index, prediction))
        for group in by_label.values():
            group_predictions = [item[1] for item in group]
            group_kept = nms_keep_indices(group_predictions, threshold)
            kept.extend(group_predictions[index] for index in group_kept)

    kept.sort(key=lambda item: float(item["score"]), reverse=True)
    if max_detections > 0:
        kept = kept[:max_detections]
    return kept


def resolve_amp_dtype(mixed_precision: str, device: torch.device) -> torch.dtype | None:
    mode = str(mixed_precision).lower()
    if mode == "auto":
        mode = "fp16" if device.type == "cuda" else "none"
    if mode in {"none", "false", "off"}:
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    raise ValueError("mixed_precision must be auto, none, fp16, or bf16")


def resolve_device(configured_device: str) -> torch.device:
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured_device)


def find_adapter_dir(root: Path) -> Path | None:
    candidates = sorted(path.parent for path in root.rglob("adapter_config.json") if path.is_file())
    return candidates[0] if candidates else None


def unpack_adapter_archive(teacher: TeacherSpec, output_dir: Path) -> Path:
    if teacher.adapter_archive is None:
        raise ValueError("adapter_archive is not set")
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from archive_unpack import archive_unpack

    unpack_dir = output_dir / "_unpacked_adapters" / safe_file_stem(teacher.name)
    cfg = OmegaConf.create(
        {
            "input_archive": str(teacher.adapter_archive),
            "output_dir": str(unpack_dir),
            "overwrite": True,
            "strip_components": 0,
            "safe_extract": True,
            "verify_manifest": True,
            "dry_run": False,
            "zstd_binary": "auto",
            "max_summary_files": 20,
        }
    )
    summary = archive_unpack(cfg)
    LOGGER.info("unpacked adapter archive for %s: %s", teacher.name, json.dumps(summary, ensure_ascii=True))
    adapter_dir = find_adapter_dir(unpack_dir)
    if adapter_dir is None:
        raise FileNotFoundError(f"No adapter_config.json found after unpacking {teacher.adapter_archive}")
    return adapter_dir


def load_teacher_model(
    teacher: TeacherSpec,
    *,
    output_dir: Path,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    processor_source = str(teacher.processor_dir) if teacher.processor_dir is not None else teacher.base_model
    processor = AutoProcessor.from_pretrained(processor_source)
    base_model = AutoModelForZeroShotObjectDetection.from_pretrained(teacher.base_model)

    adapter_source: str | None = None
    adapter_dir = teacher.adapter_dir
    if teacher.adapter_archive is not None:
        adapter_dir = unpack_adapter_archive(teacher, output_dir)
    if adapter_dir is not None:
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter_dir does not contain adapter_config.json: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        adapter_source = str(adapter_dir)
    else:
        model = base_model

    model.to(device)
    model.eval()
    info = {
        "name": teacher.name,
        "task": teacher.task,
        "base_model": teacher.base_model,
        "adapter_dir": adapter_source,
        "adapter_archive": str(teacher.adapter_archive) if teacher.adapter_archive is not None else None,
        "processor_source": processor_source,
        "queries_file": str(teacher.queries_file),
        "query_count": len(teacher.queries),
        "threshold": teacher.threshold,
        "text_threshold": teacher.text_threshold,
        "nms_threshold": teacher.nms_threshold,
        "nms_mode": teacher.nms_mode,
        "max_detections_per_image": teacher.max_detections_per_image,
    }
    return model, processor, info


def tensor_or_list_item(values: Any, index: int) -> Any:
    if isinstance(values, torch.Tensor):
        return values[index].detach().cpu().item() if values.ndim == 1 else values[index].detach().cpu().tolist()
    return values[index]


def query_for_text_label(text_label: str, queries: list[str]) -> str:
    normalized_text = normalize_query(text_label)
    for query in queries:
        if normalize_query(query) == normalized_text:
            return query
    for query in queries:
        normalized_query = normalize_query(query)
        if normalized_query in normalized_text or normalized_text in normalized_query:
            return query
    return text_label.strip().strip(".")


def postprocess_result_to_predictions(
    *,
    result: dict[str, Any],
    teacher: TeacherSpec,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels", [])
    count = len(scores)
    predictions: list[dict[str, Any]] = []
    for index in range(count):
        score = float(tensor_or_list_item(scores, index))
        box_value = tensor_or_list_item(boxes, index)
        if isinstance(box_value, torch.Tensor):
            box_value = box_value.detach().cpu().tolist()
        if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
            continue
        clipped_box = clip_box_xyxy([float(value) for value in box_value], width, height)
        if clipped_box is None:
            continue

        label_value = tensor_or_list_item(labels, index) if len(labels) > index else ""
        if isinstance(label_value, (int, float)):
            label_index = int(label_value)
            text_label = teacher.queries[label_index] if 0 <= label_index < len(teacher.queries) else str(label_value)
        else:
            text_label = str(label_value).strip()
        if not text_label:
            continue
        query = query_for_text_label(text_label, teacher.queries)
        predictions.append(
            {
                "task": teacher.task,
                "label": normalize_label(text_label),
                "query": query,
                "bbox_xyxy": [round(value, 3) for value in clipped_box],
                "score": round(score, 6),
                "teacher": teacher.name,
            }
        )
    return apply_nms(
        predictions,
        threshold=teacher.nms_threshold,
        mode=teacher.nms_mode,
        max_detections=teacher.max_detections_per_image,
    )


@torch.inference_mode()
def infer_one_image(
    *,
    model: Any,
    processor: Any,
    teacher: TeacherSpec,
    image_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[list[dict[str, Any]], int, int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    prompt = prompt_from_queries(teacher.queries)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=input_ids,
        threshold=teacher.threshold,
        text_threshold=teacher.text_threshold,
        target_sizes=[(height, width)],
        text_labels=[teacher.queries],
    )
    predictions = postprocess_result_to_predictions(
        result=results[0],
        teacher=teacher,
        width=width,
        height=height,
    )
    return predictions, width, height


def safe_file_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def overlay_path_for_image(image_item: ImageItem, *, output_dir: Path, index: int) -> Path:
    raw = Path(image_item.image_value)
    if raw.suffix.lower() in IMAGE_EXTENSIONS and not raw.is_absolute():
        return output_dir / "overlays" / raw.with_suffix(".jpg")
    return output_dir / "overlays" / f"image_{index:06d}.jpg"


def draw_overlay(
    *,
    image_item: ImageItem,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    image = Image.open(image_item.path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    colors = [
        (255, 80, 80),
        (80, 200, 255),
        (80, 230, 120),
        (255, 190, 60),
        (190, 120, 255),
    ]
    for prediction in row.get("predictions", []):
        teacher_hash = sum(ord(char) for char in str(prediction.get("teacher", "")))
        color = colors[teacher_hash % len(colors)]
        x1, y1, x2, y2 = prediction["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{prediction.get('label', '')} {float(prediction.get('score', 0.0)):.2f}"
        text_xy = (x1, max(0.0, y1 - 12.0))
        draw.text(text_xy, label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def teacher_dry_run_summary(teachers: list[TeacherSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": teacher.name,
            "task": teacher.task,
            "base_model": teacher.base_model,
            "adapter_dir": str(teacher.adapter_dir) if teacher.adapter_dir is not None else None,
            "adapter_archive": str(teacher.adapter_archive) if teacher.adapter_archive is not None else None,
            "processor_dir": str(teacher.processor_dir) if teacher.processor_dir is not None else None,
            "queries_file": str(teacher.queries_file),
            "query_count": len(teacher.queries),
            "queries": teacher.queries,
            "threshold": teacher.threshold,
            "text_threshold": teacher.text_threshold,
            "nms_threshold": teacher.nms_threshold,
            "nms_mode": teacher.nms_mode,
            "max_detections_per_image": teacher.max_detections_per_image,
        }
        for teacher in teachers
    ]


def rows_for_images(images: list[ImageItem], *, round_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in images:
        width, height = read_image_size(image.path)
        rows.append(
            {
                "image": image.image_value,
                "absolute_image_path": str(image.path),
                "width": width,
                "height": height,
                "round": round_index,
                "teacher_models": {},
                "predictions": [],
            }
        )
    return rows


def build_manifest(
    *,
    cfg: DictConfig,
    images: list[ImageItem],
    image_source: dict[str, Any],
    teacher_manifest_path: Path,
    teacher_manifest_payload: dict[str, Any],
    teachers_info: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    prediction_count = 0
    for row in rows:
        for prediction in row.get("predictions", []):
            prediction_count += 1
            task_counts[str(prediction.get("task", ""))] += 1
            label_counts[str(prediction.get("label", ""))] += 1
            teacher_counts[str(prediction.get("teacher", ""))] += 1
    return {
        "created_at": now_iso(),
        "dry_run": dry_run,
        "round": int(cfg.round),
        "image_count": len(images),
        "prediction_count": prediction_count,
        "task_counts": dict(task_counts),
        "label_counts": dict(label_counts),
        "teacher_counts": dict(teacher_counts),
        "teachers": teachers_info,
        "teacher_manifest": str(teacher_manifest_path),
        "teacher_manifest_payload": teacher_manifest_payload,
        "image_source": image_source,
        "output_dir": str(Path(to_absolute_path(str(cfg.output_dir))).resolve()),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def run_inference(cfg: DictConfig) -> dict[str, Any]:
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    images, image_source = collect_images(cfg)
    teacher_manifest_path, teacher_manifest_payload, teachers = parse_teacher_manifest(cfg)

    if bool(cfg.dry_run):
        summary = {
            "status": "dry_run",
            "image_count": len(images),
            "image_source": image_source,
            "preview_images": [str(item.path) for item in images[: int(cfg.max_preview_images)]],
            "teacher_manifest": str(teacher_manifest_path),
            "teachers": teacher_dry_run_summary(teachers),
            "output_dir": str(output_dir),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg.device))
    amp_dtype = resolve_amp_dtype(str(cfg.mixed_precision), device)
    rows = rows_for_images(images, round_index=int(cfg.round))
    teacher_infos: list[dict[str, Any]] = []

    for teacher in teachers:
        LOGGER.info("loading teacher %s (%s)", teacher.name, teacher.task)
        model, processor, teacher_info = load_teacher_model(teacher, output_dir=output_dir, device=device)
        teacher_infos.append(teacher_info)
        try:
            for index, image_item in enumerate(images):
                predictions, width, height = infer_one_image(
                    model=model,
                    processor=processor,
                    teacher=teacher,
                    image_path=image_item.path,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                rows[index]["width"] = width
                rows[index]["height"] = height
                rows[index]["teacher_models"][teacher.name] = {
                    "task": teacher.task,
                    "base_model": teacher.base_model,
                    "adapter_dir": teacher_info.get("adapter_dir"),
                    "adapter_archive": teacher_info.get("adapter_archive"),
                    "queries_file": str(teacher.queries_file),
                }
                rows[index]["predictions"].extend(predictions)
                if (index + 1) % max(1, int(cfg.log_every_images)) == 0:
                    LOGGER.info("teacher %s inferred %s/%s images", teacher.name, index + 1, len(images))
        finally:
            del model
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if bool(cfg.write_overlays):
        for index, (image_item, row) in enumerate(zip(images, rows, strict=True), start=1):
            draw_overlay(
                image_item=image_item,
                row=row,
                output_path=overlay_path_for_image(image_item, output_dir=output_dir, index=index),
            )

    write_jsonl(output_dir / "raw_predictions.jsonl", rows)
    manifest = build_manifest(
        cfg=cfg,
        images=images,
        image_source=image_source,
        teacher_manifest_path=teacher_manifest_path,
        teacher_manifest_payload=teacher_manifest_payload,
        teachers_info=teacher_infos,
        rows=rows,
        dry_run=False,
    )
    write_json(output_dir / "manifest.json", manifest)
    result = {
        "status": "ok",
        "output_dir": str(output_dir),
        "raw_predictions": str(output_dir / "raw_predictions.jsonl"),
        "manifest": str(output_dir / "manifest.json"),
        "image_count": manifest["image_count"],
        "prediction_count": manifest["prediction_count"],
        "task_counts": manifest["task_counts"],
        "label_counts": manifest["label_counts"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="run_gdino_inference",
)
def main(cfg: DictConfig) -> None:
    run_inference(cfg)


if __name__ == "__main__":
    main()
