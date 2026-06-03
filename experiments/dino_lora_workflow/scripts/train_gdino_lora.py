"""Overview:
Fine-tune Grounding DINO with LoRA from a guardrail dataset and optional reviewed pseudo labels.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py guardrail_dir=data/dino_workflow/guardrail/current output_dir=outputs/dino/training/example
    .venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py dry_run=true skip_model_load=true guardrail_dir=outputs/tmp/guardrail

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/train_gdino_lora.yaml`.
    - Each input directory must contain `annotations.jsonl` or, for pseudo data, `selected_annotations.jsonl`, plus image files.
    - Guardrail rows are treated as trusted data; pseudo rows are down-weighted by `pseudo_loss_weight`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


LOGGER = logging.getLogger("dino_lora_train")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class DetectionSample:
    """One image-level Grounding DINO sample."""

    image_path: Path
    width: int
    height: int
    annotations: list[dict[str, Any]]
    dataset_kind: str
    sample_weight: float
    split: str
    row_index: int
    source_file: Path


class JsonlDetectionDataset(Dataset):
    """Dataset that emits PIL images, prompts, normalized boxes, and sample weights."""

    def __init__(self, samples: list[DetectionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        queries = unique_queries(sample.annotations)
        query_to_index = {query: query_index for query_index, query in enumerate(queries)}

        boxes: list[list[float]] = []
        class_labels: list[int] = []
        kept_annotations: list[dict[str, Any]] = []
        for annotation in sample.annotations:
            query = annotation_query(annotation)
            bbox_xyxy = annotation_bbox_xyxy(annotation)
            if query is None or bbox_xyxy is None:
                continue
            normalized = normalize_xyxy_to_cxcywh(bbox_xyxy, sample.width, sample.height)
            if normalized is None:
                continue
            boxes.append(normalized)
            class_labels.append(query_to_index[query])
            kept_annotations.append(annotation)

        if not boxes:
            raise ValueError(f"Sample unexpectedly has no valid boxes: {sample.image_path}")

        prompt = prompt_from_queries(queries)
        return {
            "image": image,
            "text": prompt,
            "label": {
                "class_labels": torch.tensor(class_labels, dtype=torch.long),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
            },
            "sample_weight": torch.tensor(float(sample.sample_weight), dtype=torch.float32),
            "dataset_kind": sample.dataset_kind,
            "image_path": str(sample.image_path),
            "annotation_count": len(kept_annotations),
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_plain_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(to_absolute_path(text)).resolve()


def resolve_dir(value: Any, *, field_name: str, required: bool = True) -> Path | None:
    path = optional_path(value)
    if path is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if not path.is_dir():
        raise FileNotFoundError(f"{field_name} is not a directory: {path}")
    return path


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


def configure_logging(output_dir: Path, dry_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.disabled = False
    LOGGER.propagate = False
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    if not dry_run:
        file_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def candidate_annotation_files(dataset_dir: Path, preferred_name: str | None) -> list[Path]:
    candidates: list[Path] = []
    if preferred_name:
        candidates.append(dataset_dir / preferred_name)
    candidates.extend([dataset_dir / "annotations.jsonl", dataset_dir / "selected_annotations.jsonl"])
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def find_annotation_file(dataset_dir: Path, *, dataset_kind: str) -> Path:
    preferred = "selected_annotations.jsonl" if dataset_kind == "pseudo" else "annotations.jsonl"
    for path in candidate_annotation_files(dataset_dir, preferred):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"No annotations JSONL found in {dataset_dir}. Expected annotations.jsonl"
        + (" or selected_annotations.jsonl" if dataset_kind == "pseudo" else "")
    )


def resolve_image_path(image_value: Any, *, dataset_dir: Path, source_file: Path) -> Path:
    if image_value is None or not str(image_value).strip():
        raise ValueError(f"Row in {source_file} does not include image or image_path")

    raw = Path(str(image_value)).expanduser()
    if raw.is_absolute() and raw.is_file():
        return raw.resolve()

    candidates = [
        (dataset_dir / raw).resolve(),
        (source_file.parent / raw).resolve(),
        (dataset_dir / "images" / raw).resolve(),
        (source_file.parent / "images" / raw).resolve(),
        (dataset_dir / "images" / raw.name).resolve(),
        (source_file.parent / "images" / raw.name).resolve(),
        Path(to_absolute_path(str(raw))).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def normalize_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = row.get("annotations")
    if isinstance(annotations, list):
        return [dict(item) for item in annotations if isinstance(item, dict)]
    if isinstance(annotations, dict):
        return [dict(annotations)]

    row_level_excludes = {
        "image",
        "image_path",
        "absolute_image_path",
        "width",
        "height",
        "split",
        "label_source",
        "weight",
        "source",
        "guardrail",
        "source_annotation_file",
        "original_image_path",
    }
    annotation = {key: value for key, value in row.items() if key not in row_level_excludes}
    return [annotation] if annotation else []


def annotation_query(annotation: dict[str, Any]) -> str | None:
    for key in ("query", "text", "prompt"):
        value = annotation.get(key)
        if value is not None and str(value).strip():
            return normalize_query(str(value))
    label = annotation.get("label")
    if label is not None and str(label).strip():
        return normalize_query(str(label).replace("_", " "))
    return None


def normalize_query(value: str) -> str:
    return value.strip().strip(".").strip().lower()


def prompt_from_queries(queries: list[str]) -> str:
    return " ".join(f"{query}." for query in queries)


def unique_queries(annotations: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for annotation in annotations:
        query = annotation_query(annotation)
        if query is None or query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def annotation_bbox_xyxy(annotation: dict[str, Any]) -> list[float] | None:
    value = annotation.get("bbox_xyxy")
    if value is None:
        value = annotation.get("box_xyxy")
    if value is None and annotation.get("bbox") is not None:
        value = annotation.get("bbox")
        fmt = str(annotation.get("bbox_format") or annotation.get("box_format") or "xyxy").lower()
        if fmt in {"xywh", "coco"}:
            x, y, width, height = [float(item) for item in value]
            return [x, y, x + width, y + height]
        if fmt == "cxcywh":
            cx, cy, width, height = [float(item) for item in value]
            return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]
    if value is None and all(key in annotation for key in ("x1", "y1", "x2", "y2")):
        value = [annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]]
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def normalize_xyxy_to_cxcywh(bbox_xyxy: list[float], width: int, height: int) -> list[float] | None:
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    return [
        ((x1 + x2) / 2.0) / float(width),
        ((y1 + y2) / 2.0) / float(height),
        box_width / float(width),
        box_height / float(height),
    ]


def row_effective_weight(
    row: dict[str, Any],
    *,
    dataset_kind: str,
    pseudo_loss_weight: float,
    respect_row_weight: bool,
) -> float:
    row_weight = 1.0
    if respect_row_weight and row.get("weight") is not None:
        try:
            row_weight = float(row["weight"])
        except (TypeError, ValueError):
            row_weight = 1.0
    if dataset_kind == "pseudo":
        row_weight *= float(pseudo_loss_weight)
    return row_weight


def normalize_split(value: Any) -> str:
    if value is None or not str(value).strip():
        return "train"
    return str(value).strip().lower()


def load_samples(
    dataset_dir: Path,
    *,
    dataset_kind: str,
    pseudo_loss_weight: float,
    respect_row_weight: bool,
) -> tuple[list[DetectionSample], dict[str, Any]]:
    annotation_file = find_annotation_file(dataset_dir, dataset_kind=dataset_kind)
    rows = read_jsonl(annotation_file)

    samples: list[DetectionSample] = []
    skipped_empty = 0
    skipped_missing_image = 0
    skipped_bad_box = 0
    label_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        annotations = normalize_annotations(row)
        valid_annotations: list[dict[str, Any]] = []
        width = row.get("width")
        height = row.get("height")
        image_value = row.get("image", row.get("image_path"))
        image_path = resolve_image_path(image_value, dataset_dir=dataset_dir, source_file=annotation_file)
        if not image_path.is_file():
            skipped_missing_image += 1
            continue
        if width is None or height is None:
            width, height = read_image_size(image_path)
        width = int(width)
        height = int(height)

        for annotation in annotations:
            query = annotation_query(annotation)
            bbox_xyxy = annotation_bbox_xyxy(annotation)
            if query is None or bbox_xyxy is None:
                skipped_bad_box += 1
                continue
            if normalize_xyxy_to_cxcywh(bbox_xyxy, width, height) is None:
                skipped_bad_box += 1
                continue
            valid_annotations.append(annotation)
            label_counts[str(annotation.get("label") or query)] += 1
            if annotation.get("task") is not None:
                task_counts[str(annotation["task"])] += 1
            query_counts[query] += 1

        if not valid_annotations:
            skipped_empty += 1
            continue

        samples.append(
            DetectionSample(
                image_path=image_path,
                width=width,
                height=height,
                annotations=valid_annotations,
                dataset_kind=dataset_kind,
                sample_weight=row_effective_weight(
                    row,
                    dataset_kind=dataset_kind,
                    pseudo_loss_weight=pseudo_loss_weight,
                    respect_row_weight=respect_row_weight,
                ),
                split=normalize_split(row.get("split")),
                row_index=row_index,
                source_file=annotation_file,
            )
        )

    summary = {
        "dataset_dir": str(dataset_dir),
        "annotation_file": str(annotation_file),
        "rows": len(rows),
        "samples": len(samples),
        "skipped_empty": skipped_empty,
        "skipped_missing_image": skipped_missing_image,
        "skipped_bad_box_or_query": skipped_bad_box,
        "label_counts": dict(label_counts),
        "task_counts": dict(task_counts),
        "query_counts": dict(query_counts),
    }
    return samples, summary


def deterministic_guardrail_split(
    samples: list[DetectionSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[DetectionSample], list[DetectionSample]]:
    explicit_val = [sample for sample in samples if sample.split == "val"]
    explicit_train = [sample for sample in samples if sample.split != "val"]
    if explicit_val:
        return explicit_train, explicit_val

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("guardrail_val_fraction must be in [0, 1)")
    if len(samples) < 2 or val_fraction <= 0.0:
        return samples, samples[: min(1, len(samples))]

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_fraction)))
    val_indices = set(indices[:val_count])
    train = [sample for index, sample in enumerate(samples) if index not in val_indices]
    val = [sample for index, sample in enumerate(samples) if index in val_indices]
    if not train:
        train = samples
    return train, val


def repeated_dataset(samples: list[DetectionSample], repeat: int) -> JsonlDetectionDataset:
    if repeat < 1:
        raise ValueError("repeat values must be >= 1")
    return JsonlDetectionDataset(samples * repeat)


def build_train_dataset(
    *,
    guardrail_train: list[DetectionSample],
    pseudo_train: list[DetectionSample],
    guardrail_repeat: int,
    pseudo_repeat: int,
) -> Dataset:
    datasets: list[Dataset] = [repeated_dataset(guardrail_train, guardrail_repeat)]
    if pseudo_train:
        datasets.append(repeated_dataset(pseudo_train, pseudo_repeat))
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_sampler(
    dataset: Dataset,
    *,
    guardrail_fraction: float | None,
    seed: int,
) -> WeightedRandomSampler | None:
    if guardrail_fraction is None:
        return None
    if not isinstance(dataset, ConcatDataset) or len(dataset.datasets) != 2:
        return None
    if not 0.0 < guardrail_fraction < 1.0:
        raise ValueError("guardrail_fraction must be in (0, 1) when set")

    guardrail_len = len(dataset.datasets[0])
    pseudo_len = len(dataset.datasets[1])
    guardrail_weight = guardrail_fraction / max(1, guardrail_len)
    pseudo_weight = (1.0 - guardrail_fraction) / max(1, pseudo_len)
    weights = [guardrail_weight] * guardrail_len + [pseudo_weight] * pseudo_len
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "texts": [item["text"] for item in batch],
        "labels": [item["label"] for item in batch],
        "sample_weights": torch.stack([item["sample_weight"] for item in batch]),
        "dataset_kinds": [item["dataset_kind"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "annotation_counts": [item["annotation_count"] for item in batch],
    }


def move_labels_to_device(labels: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in label.items()} for label in labels]


def make_inputs(processor: Any, batch: dict[str, Any], device: torch.device) -> tuple[Any, list[dict[str, torch.Tensor]]]:
    inputs = processor(images=batch["images"], text=batch["texts"], return_tensors="pt")
    return inputs.to(device), move_labels_to_device(batch["labels"], device)


def lora_target_modules(model: nn.Module) -> list[str]:
    allowed_tokens = (
        ".self_attn.",
        ".encoder_attn_text.",
        ".encoder_attn.",
        ".fusion_layer.attn.",
        ".text_enhancer_layer.self_attn.",
    )
    allowed_names = (
        "query",
        "key",
        "value",
        "out_proj",
        "vision_proj",
        "text_proj",
        "values_vision_proj",
        "values_text_proj",
        "out_vision_proj",
        "out_text_proj",
        "value_proj",
        "output_proj",
    )
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not (name.startswith("model.encoder.") or name.startswith("model.decoder.")):
            continue
        if any(token in name for token in allowed_tokens) and name.rsplit(".", 1)[-1] in allowed_names:
            targets.append(name)
    if not targets:
        raise RuntimeError("No LoRA target modules found")
    return sorted(targets)


def patch_input_embeddings_for_peft(model: nn.Module) -> None:
    text_backbone = model.model.text_backbone
    embeddings = text_backbone.embeddings.word_embeddings

    def get_input_embeddings() -> nn.Module:
        return embeddings

    model.get_input_embeddings = get_input_embeddings  # type: ignore[method-assign]
    model.model.get_input_embeddings = get_input_embeddings  # type: ignore[method-assign]


def freeze_model_parts(model: nn.Module, *, freeze_vision_backbone: bool, freeze_text_backbone: bool) -> None:
    if freeze_vision_backbone:
        if hasattr(model.model, "freeze_backbone"):
            model.model.freeze_backbone()
        for parameter in model.model.backbone.parameters():
            parameter.requires_grad = False
    if freeze_text_backbone:
        for parameter in model.model.text_backbone.parameters():
            parameter.requires_grad = False


def resolve_device(configured_device: str) -> torch.device:
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured_device)


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


def unpack_adapter_archive(archive_path: Path, output_dir: Path, configured_unpack_dir: Path | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from archive_unpack import archive_unpack

    unpack_dir = configured_unpack_dir or (output_dir / "_resume_adapter_unpacked")
    cfg = OmegaConf.create(
        {
            "input_archive": str(archive_path),
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
    LOGGER.info("unpacked resume adapter archive: %s", json.dumps(summary, ensure_ascii=True))
    adapter_dir = find_adapter_dir(unpack_dir)
    if adapter_dir is None:
        raise FileNotFoundError(f"No PEFT adapter_config.json found after unpacking {archive_path}")
    return adapter_dir


def find_adapter_dir(root: Path) -> Path | None:
    candidates = sorted(path.parent for path in root.rglob("adapter_config.json") if path.is_file())
    return candidates[0] if candidates else None


def maybe_load_resume_adapter(cfg: DictConfig, output_dir: Path) -> Path | None:
    resume_adapter_dir = optional_path(cfg.resume_adapter_dir)
    resume_adapter_archive = optional_path(cfg.resume_adapter_archive)
    if resume_adapter_dir is not None and resume_adapter_archive is not None:
        raise ValueError("Set only one of resume_adapter_dir or resume_adapter_archive")
    if resume_adapter_dir is not None:
        if not (resume_adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"resume_adapter_dir does not contain adapter_config.json: {resume_adapter_dir}")
        return resume_adapter_dir
    if resume_adapter_archive is not None:
        if not resume_adapter_archive.is_file():
            raise FileNotFoundError(f"resume_adapter_archive not found: {resume_adapter_archive}")
        return unpack_adapter_archive(resume_adapter_archive, output_dir, optional_path(cfg.resume_adapter_unpack_dir))
    return None


def build_model_and_processor(
    cfg: DictConfig,
    *,
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, Any, list[str], dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(str(cfg.base_model))
    base_model = AutoModelForZeroShotObjectDetection.from_pretrained(str(cfg.base_model))
    freeze_model_parts(
        base_model,
        freeze_vision_backbone=bool(cfg.freeze_vision_backbone),
        freeze_text_backbone=bool(cfg.freeze_text_backbone),
    )
    target_modules = lora_target_modules(base_model)
    patch_input_embeddings_for_peft(base_model)

    resume_adapter_dir = maybe_load_resume_adapter(cfg, output_dir)
    if resume_adapter_dir is not None:
        model = PeftModel.from_pretrained(base_model, resume_adapter_dir, is_trainable=True)
        adapter_source = str(resume_adapter_dir)
    else:
        lora_cfg = LoraConfig(
            r=int(cfg.lora.r),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            bias=str(cfg.lora.bias),
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_cfg)
        adapter_source = None

    model.to(device)
    model.train()
    model_info = {
        "lora_target_modules": target_modules,
        "resume_adapter_dir": adapter_source,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, processor, target_modules, model_info


def batch_loss(
    model: nn.Module,
    processor: Any,
    batch: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    inputs, labels = make_inputs(processor, batch, device)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        sample_weight = batch["sample_weights"].to(device).mean()
        return loss * sample_weight


@torch.no_grad()
def evaluate_guardrail(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int,
) -> float:
    if len(loader) == 0 or max_batches <= 0:
        return float("nan")
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        loss = batch_loss(model, processor, batch, device, amp_dtype)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "elapsed_sec",
                "guardrail_batches",
                "pseudo_batches",
            ],
        )
        writer.writeheader()


def append_metric(
    path: Path,
    *,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    elapsed_sec: float,
    guardrail_batches: int,
    pseudo_batches: int,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "elapsed_sec",
                "guardrail_batches",
                "pseudo_batches",
            ],
        )
        writer.writerow(
            {
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "elapsed_sec": elapsed_sec,
                "guardrail_batches": guardrail_batches,
                "pseudo_batches": pseudo_batches,
            }
        )


def optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    clip_grad_norm: float,
) -> None:
    if scaler is not None:
        if clip_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def compute_effective_max_steps(cfg: DictConfig, train_loader: DataLoader) -> int:
    if int(cfg.smoke_max_steps) > 0:
        return int(cfg.smoke_max_steps)
    if int(cfg.max_steps) > 0:
        return int(cfg.max_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, int(cfg.grad_accum))))
    return max(1, int(cfg.epochs) * steps_per_epoch)


def count_batch_sources(batch: dict[str, Any]) -> tuple[int, int]:
    guardrail_count = sum(1 for kind in batch["dataset_kinds"] if kind == "guardrail")
    pseudo_count = sum(1 for kind in batch["dataset_kinds"] if kind == "pseudo")
    return guardrail_count, pseudo_count


def train_loop(
    cfg: DictConfig,
    *,
    model: nn.Module,
    processor: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> None:
    metrics_path = output_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    scaler: torch.amp.GradScaler | None = None
    if amp_dtype == torch.float16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    max_steps = compute_effective_max_steps(cfg, train_loader)
    LOGGER.info("training for %s optimizer steps", max_steps)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_items = 0
    running_guardrail = 0
    running_pseudo = 0
    epoch = 0
    grad_accum = max(1, int(cfg.grad_accum))

    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            loss = batch_loss(model, processor, batch, device, amp_dtype)
            scaled_loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_step += 1
            running_loss += float(loss.detach().cpu())
            running_items += 1
            guardrail_batch_count, pseudo_batch_count = count_batch_sources(batch)
            running_guardrail += guardrail_batch_count
            running_pseudo += pseudo_batch_count

            if micro_step % grad_accum != 0:
                continue

            optimizer_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                clip_grad_norm=float(cfg.clip_grad_norm),
            )
            global_step += 1
            train_loss = running_loss / max(1, running_items)

            should_eval = (
                global_step == 1
                or global_step % max(1, int(cfg.eval_every_steps)) == 0
                or global_step == max_steps
            )
            if should_eval:
                val_loss = evaluate_guardrail(
                    model,
                    processor,
                    val_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    max_batches=int(cfg.eval_batches),
                )
                append_metric(
                    metrics_path,
                    step=global_step,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    elapsed_sec=time.time() - started,
                    guardrail_batches=running_guardrail,
                    pseudo_batches=running_pseudo,
                )
                LOGGER.info(
                    "step=%s/%s epoch=%s train_loss=%.5f val_loss=%.5f",
                    global_step,
                    max_steps,
                    epoch,
                    train_loss,
                    val_loss,
                )
                running_loss = 0.0
                running_items = 0
                running_guardrail = 0
                running_pseudo = 0

            if global_step >= max_steps:
                break

    if micro_step % grad_accum != 0:
        optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad_norm=float(cfg.clip_grad_norm),
        )


def archive_adapter(output_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from archive_pack import archive_pack

    cfg = OmegaConf.create(
        {
            "input_paths": [str(output_dir / "adapter")],
            "output_archive": str(output_dir / "adapter.tar.zst"),
            "base_dir": str(output_dir),
            "include_manifest": True,
            "exclude_globs": ["__pycache__", "__pycache__/**", "*.pyc", ".DS_Store"],
            "overwrite": overwrite,
            "dry_run": False,
            "compression_level": 3,
            "zstd_binary": "auto",
            "max_summary_files": 20,
        }
    )
    return archive_pack(cfg)


def build_dataloaders(
    cfg: DictConfig,
    *,
    train_dataset: Dataset,
    guardrail_val: list[DetectionSample],
    sampler: WeightedRandomSampler | None,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
        pin_memory=resolve_device(str(cfg.device)).type == "cuda",
    )
    val_loader = DataLoader(
        JsonlDetectionDataset(guardrail_val),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
        pin_memory=resolve_device(str(cfg.device)).type == "cuda",
    )
    return train_loader, val_loader


def dataset_preview(samples: list[DetectionSample], max_items: int) -> list[dict[str, Any]]:
    preview = []
    for sample in samples[:max_items]:
        preview.append(
            {
                "image_path": str(sample.image_path),
                "width": sample.width,
                "height": sample.height,
                "dataset_kind": sample.dataset_kind,
                "split": sample.split,
                "sample_weight": sample.sample_weight,
                "annotation_count": len(sample.annotations),
                "queries": unique_queries(sample.annotations),
            }
        )
    return preview


def collect_datasets(cfg: DictConfig) -> dict[str, Any]:
    guardrail_dir = resolve_dir(cfg.guardrail_dir, field_name="guardrail_dir", required=True)
    assert guardrail_dir is not None
    pseudo_dir = resolve_dir(cfg.pseudo_dir, field_name="pseudo_dir", required=False)

    guardrail_samples, guardrail_summary = load_samples(
        guardrail_dir,
        dataset_kind="guardrail",
        pseudo_loss_weight=float(cfg.pseudo_loss_weight),
        respect_row_weight=bool(cfg.respect_row_weight),
    )
    if not guardrail_samples:
        raise ValueError(f"No valid guardrail samples found in {guardrail_dir}")

    pseudo_samples: list[DetectionSample] = []
    pseudo_summary: dict[str, Any] | None = None
    if pseudo_dir is not None:
        pseudo_samples, pseudo_summary = load_samples(
            pseudo_dir,
            dataset_kind="pseudo",
            pseudo_loss_weight=float(cfg.pseudo_loss_weight),
            respect_row_weight=bool(cfg.respect_row_weight),
        )

    guardrail_train, guardrail_val = deterministic_guardrail_split(
        guardrail_samples,
        val_fraction=float(cfg.guardrail_val_fraction),
        seed=int(cfg.seed),
    )
    if bool(cfg.include_val_in_train):
        guardrail_train = guardrail_samples

    pseudo_train = [sample for sample in pseudo_samples if sample.split != "val"]
    train_dataset = build_train_dataset(
        guardrail_train=guardrail_train,
        pseudo_train=pseudo_train,
        guardrail_repeat=int(cfg.guardrail_repeat),
        pseudo_repeat=int(cfg.pseudo_repeat),
    )
    sampler = build_sampler(
        train_dataset,
        guardrail_fraction=None if cfg.guardrail_fraction is None else float(cfg.guardrail_fraction),
        seed=int(cfg.seed),
    )
    return {
        "guardrail_samples": guardrail_samples,
        "guardrail_train": guardrail_train,
        "guardrail_val": guardrail_val,
        "pseudo_samples": pseudo_samples,
        "pseudo_train": pseudo_train,
        "train_dataset": train_dataset,
        "sampler": sampler,
        "guardrail_summary": guardrail_summary,
        "pseudo_summary": pseudo_summary,
    }


def write_metadata(
    output_dir: Path,
    *,
    cfg: DictConfig,
    dataset_info: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    model_info: dict[str, Any] | None,
    dry_run_summary: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "created_at": now_iso(),
        "config": to_plain_container(cfg),
        "device": str(device),
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
        "guardrail": dataset_info["guardrail_summary"],
        "pseudo": dataset_info["pseudo_summary"],
        "counts": {
            "guardrail_samples": len(dataset_info["guardrail_samples"]),
            "guardrail_train": len(dataset_info["guardrail_train"]),
            "guardrail_val": len(dataset_info["guardrail_val"]),
            "pseudo_samples": len(dataset_info["pseudo_samples"]),
            "pseudo_train": len(dataset_info["pseudo_train"]),
            "train_dataset_length": len(dataset_info["train_dataset"]),
            "uses_weighted_sampler": dataset_info["sampler"] is not None,
        },
        "model": model_info,
        "dry_run": dry_run_summary,
    }
    write_json(output_dir / "metadata.json", metadata)


def train_from_config(cfg: DictConfig) -> dict[str, Any]:
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    configure_logging(output_dir, bool(cfg.dry_run))

    if int(cfg.batch_size) < 1 or int(cfg.grad_accum) < 1:
        raise ValueError("batch_size and grad_accum must be positive")
    if float(cfg.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(cfg.pseudo_loss_weight) <= 0.0:
        raise ValueError("pseudo_loss_weight must be positive")

    dataset_info = collect_datasets(cfg)
    dry_run_summary = {
        "preview": dataset_preview(
            dataset_info["guardrail_samples"] + dataset_info["pseudo_samples"],
            int(cfg.max_preview_samples),
        )
    }
    if bool(cfg.dry_run):
        write_metadata(
            output_dir,
            cfg=cfg,
            dataset_info=dataset_info,
            device=resolve_device(str(cfg.device)),
            amp_dtype=None,
            model_info=None,
            dry_run_summary=dry_run_summary,
        )
        summary = {
            "status": "dry_run",
            "output_dir": str(output_dir),
            "counts": {
                "guardrail_samples": len(dataset_info["guardrail_samples"]),
                "guardrail_train": len(dataset_info["guardrail_train"]),
                "guardrail_val": len(dataset_info["guardrail_val"]),
                "pseudo_samples": len(dataset_info["pseudo_samples"]),
                "pseudo_train": len(dataset_info["pseudo_train"]),
                "train_dataset_length": len(dataset_info["train_dataset"]),
            },
            "preview": dry_run_summary["preview"],
        }
        LOGGER.info("dry run summary: %s", json.dumps(summary, ensure_ascii=True))
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    if bool(cfg.skip_model_load):
        raise ValueError("skip_model_load=true is only valid with dry_run=true")

    device = resolve_device(str(cfg.device))
    amp_dtype = resolve_amp_dtype(str(cfg.mixed_precision), device)
    train_loader, val_loader = build_dataloaders(
        cfg,
        train_dataset=dataset_info["train_dataset"],
        guardrail_val=dataset_info["guardrail_val"],
        sampler=dataset_info["sampler"],
    )
    model, processor, _target_modules, model_info = build_model_and_processor(
        cfg,
        output_dir=output_dir,
        device=device,
    )
    LOGGER.info("trainable params: %s / %s", model_info["trainable_params"], model_info["total_params"])

    write_metadata(
        output_dir,
        cfg=cfg,
        dataset_info=dataset_info,
        device=device,
        amp_dtype=amp_dtype,
        model_info=model_info,
        dry_run_summary=None,
    )

    train_loop(
        cfg,
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )

    adapter_dir = output_dir / "adapter"
    processor_dir = output_dir / "processor"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(processor_dir)
    LOGGER.info("saved adapter: %s", adapter_dir)
    LOGGER.info("saved processor: %s", processor_dir)

    archive_summary = None
    if bool(cfg.save_archive):
        archive_summary = archive_adapter(output_dir, overwrite=bool(cfg.archive_overwrite))
        LOGGER.info("saved adapter archive: %s", json.dumps(archive_summary, ensure_ascii=True))

    result = {
        "status": "trained",
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "processor_dir": str(processor_dir),
        "archive": archive_summary,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_gdino_lora",
)
def main(cfg: DictConfig) -> None:
    train_from_config(cfg)


if __name__ == "__main__":
    main()
