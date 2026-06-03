"""Overview:
Apply binary review decisions to Grounding DINO raw pseudo-label predictions.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/apply_review_decisions.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/apply_review_decisions.py include_unsure_as_selected=true

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/apply_review_decisions.yaml`.
    - Image-level review decisions accept or reject every prediction stored in the review queue item.
    - Annotation-level review decisions are grouped back into one JSONL row per image for training.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


DECISIONS = {"accepted", "rejected", "unsure"}


@dataclass(slots=True)
class ImageGroup:
    """Accumulated selected or rejected annotations for one image."""

    image: str
    width: int | None
    height: int | None
    label_source: str
    pseudo_round: int | None
    weight: float
    annotations: list[dict[str, Any]] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def normalize_decision(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"accept", "accepted"}:
        return "accepted"
    if text in {"reject", "rejected"}:
        return "rejected"
    if text == "unsure":
        return "unsure"
    return None


def load_latest_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        review_id = row.get("review_id")
        decision = normalize_decision(row.get("decision"))
        if review_id is None or decision is None:
            continue
        normalized = dict(row)
        normalized["review_id"] = str(review_id)
        normalized["decision"] = decision
        decisions[str(review_id)] = normalized
    return decisions


def load_review_queue(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        review_id = str(row.get("review_id") or f"review_{index:06d}")
        if review_id in seen:
            raise ValueError(f"Duplicate review_id in queue: {review_id}")
        seen.add(review_id)
        normalized = dict(row)
        normalized["review_id"] = review_id
        queue.append(normalized)
    return queue


def raw_rows_by_line(path: Path) -> dict[int, dict[str, Any]]:
    return {index: row for index, row in enumerate(read_jsonl(path), start=1)}


def raw_ref(queue_item: dict[str, Any]) -> dict[str, Any]:
    ref = queue_item.get("raw_prediction_ref")
    return dict(ref) if isinstance(ref, dict) else {}


def raw_row_for_item(queue_item: dict[str, Any], raw_by_line: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    ref = raw_ref(queue_item)
    line_index = ref.get("line_index")
    try:
        line_number = int(line_index)
    except (TypeError, ValueError):
        return None
    return raw_by_line.get(line_number)


def image_value(queue_item: dict[str, Any], raw_row: dict[str, Any] | None) -> str:
    for candidate in (
        queue_item.get("image"),
        raw_row.get("image") if raw_row else None,
        raw_row.get("image_path") if raw_row else None,
        raw_row.get("absolute_image_path") if raw_row else None,
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    return f"missing_image_for_{queue_item['review_id']}"


def integer_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_width_height(queue_item: dict[str, Any], raw_row: dict[str, Any] | None) -> tuple[int | None, int | None]:
    width = integer_or_none(queue_item.get("width"))
    height = integer_or_none(queue_item.get("height"))
    if raw_row is not None:
        width = width if width is not None else integer_or_none(raw_row.get("width"))
        height = height if height is not None else integer_or_none(raw_row.get("height"))
    return width, height


def item_pseudo_round(queue_item: dict[str, Any], raw_row: dict[str, Any] | None, configured_round: Any) -> int | None:
    if configured_round is not None:
        return integer_or_none(configured_round)
    for value in (raw_row.get("round") if raw_row else None, queue_item.get("round")):
        result = integer_or_none(value)
        if result is not None:
            return result
    return None


def normalize_bbox(value: Any, *, width: int | None, height: int | None, clip_boxes: bool) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if clip_boxes:
        if width is not None:
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
        if height is not None:
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(value, 3) for value in (x1, y1, x2, y2)]


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().strip(".").lower()
    return "_".join(part for part in text.replace("-", " ").split() if part) or "unknown"


def query_from_prediction(prediction: dict[str, Any], label: str) -> str:
    for key in ("query", "text", "prompt"):
        value = prediction.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip(".")
    return label.replace("_", " ")


def score_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def prediction_to_annotation(
    prediction: dict[str, Any],
    *,
    review_id: str,
    raw_prediction_ref: dict[str, Any],
    width: int | None,
    height: int | None,
    clip_boxes: bool,
    include_review_metadata: bool,
) -> dict[str, Any] | None:
    bbox = normalize_bbox(
        prediction.get("bbox_xyxy") or prediction.get("box_xyxy") or prediction.get("bbox"),
        width=width,
        height=height,
        clip_boxes=clip_boxes,
    )
    if bbox is None:
        return None

    label = normalize_label(prediction.get("label") or prediction.get("query"))
    annotation: dict[str, Any] = {
        "task": str(prediction.get("task") or "").strip(),
        "label": label,
        "query": query_from_prediction(prediction, label),
        "bbox_xyxy": bbox,
        "score": score_or_none(prediction.get("score")),
        "teacher": prediction.get("teacher"),
    }
    if prediction.get("kp_index") is not None:
        annotation["kp_index"] = prediction.get("kp_index")
    if include_review_metadata:
        annotation["review_id"] = review_id
        annotation["raw_prediction_ref"] = raw_prediction_ref
    return annotation


def source_base(
    *,
    raw_predictions_file: Path,
    review_queue_file: Path,
    review_decisions_file: Path,
) -> dict[str, Any]:
    return {
        "raw_predictions_file": str(raw_predictions_file),
        "review_queue_file": str(review_queue_file),
        "review_decisions_file": str(review_decisions_file),
        "review_ids": [],
        "review_units": [],
        "raw_prediction_refs": [],
    }


def add_source_ref(group: ImageGroup, *, queue_item: dict[str, Any], ref: dict[str, Any]) -> None:
    review_id = str(queue_item["review_id"])
    if review_id not in group.source["review_ids"]:
        group.source["review_ids"].append(review_id)
    review_unit = str(queue_item.get("review_unit") or "")
    if review_unit and review_unit not in group.source["review_units"]:
        group.source["review_units"].append(review_unit)
    if ref and ref not in group.source["raw_prediction_refs"]:
        group.source["raw_prediction_refs"].append(ref)


def ensure_group(
    groups: OrderedDict[str, ImageGroup],
    *,
    queue_item: dict[str, Any],
    raw_row: dict[str, Any] | None,
    label_source: str,
    weight: float,
    configured_round: Any,
    source_template: dict[str, Any],
) -> ImageGroup:
    image = image_value(queue_item, raw_row)
    group = groups.get(image)
    if group is not None:
        return group

    width, height = row_width_height(queue_item, raw_row)
    group = ImageGroup(
        image=image,
        width=width,
        height=height,
        label_source=label_source,
        pseudo_round=item_pseudo_round(queue_item, raw_row, configured_round),
        weight=weight,
        source=json.loads(json.dumps(source_template)),
    )
    groups[image] = group
    return group


def group_to_row(group: ImageGroup) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": group.image,
        "width": group.width,
        "height": group.height,
        "label_source": group.label_source,
        "pseudo_round": group.pseudo_round,
        "weight": group.weight,
        "annotations": group.annotations,
        "source": group.source,
    }
    return row


def decision_target(decision: str, cfg: DictConfig) -> str | None:
    if decision == "accepted":
        return "selected"
    if decision == "rejected":
        return "rejected"
    if decision != "unsure":
        return None
    if bool(cfg.include_unsure_as_selected) and bool(cfg.include_unsure_as_rejected):
        raise ValueError("include_unsure_as_selected and include_unsure_as_rejected cannot both be true")
    if bool(cfg.include_unsure_as_selected):
        return "selected"
    if bool(cfg.include_unsure_as_rejected):
        return "rejected"
    return None


def predictions_from_queue_item(queue_item: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = queue_item.get("predictions")
    if not isinstance(predictions, list):
        return []
    return [dict(prediction) for prediction in predictions if isinstance(prediction, dict)]


def apply_review_decisions(cfg: DictConfig) -> dict[str, Any]:
    raw_predictions_file = Path(to_absolute_path(str(cfg.raw_predictions_file))).resolve()
    review_queue_file = Path(to_absolute_path(str(cfg.review_queue_file))).resolve()
    review_decisions_file = Path(to_absolute_path(str(cfg.review_decisions_file))).resolve()
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()

    if bool(cfg.ignore_unsure) is False and not bool(cfg.include_unsure_as_selected) and not bool(cfg.include_unsure_as_rejected):
        raise ValueError("ignore_unsure=false requires include_unsure_as_selected=true or include_unsure_as_rejected=true")

    raw_by_line = raw_rows_by_line(raw_predictions_file)
    queue = load_review_queue(review_queue_file)
    decisions = load_latest_decisions(review_decisions_file)
    queue_review_ids = {str(item["review_id"]) for item in queue}

    source_template = source_base(
        raw_predictions_file=raw_predictions_file,
        review_queue_file=review_queue_file,
        review_decisions_file=review_decisions_file,
    )
    selected_groups: OrderedDict[str, ImageGroup] = OrderedDict()
    rejected_groups: OrderedDict[str, ImageGroup] = OrderedDict()

    counts = Counter()
    selected_task_counts: Counter[str] = Counter()
    selected_label_counts: Counter[str] = Counter()
    rejected_task_counts: Counter[str] = Counter()
    rejected_label_counts: Counter[str] = Counter()
    dropped_prediction_count = 0

    for queue_item in queue:
        review_id = str(queue_item["review_id"])
        decision_row = decisions.get(review_id)
        decision = normalize_decision(decision_row.get("decision")) if decision_row is not None else None
        if decision is None:
            counts["pending"] += 1
            continue
        counts[decision] += 1

        target = decision_target(decision, cfg)
        if target is None:
            counts["ignored"] += 1
            continue

        raw_row = raw_row_for_item(queue_item, raw_by_line)
        width, height = row_width_height(queue_item, raw_row)
        ref = raw_ref(queue_item)
        group = ensure_group(
            selected_groups if target == "selected" else rejected_groups,
            queue_item=queue_item,
            raw_row=raw_row,
            label_source="pseudo_reviewed" if target == "selected" else "pseudo_rejected",
            weight=float(cfg.selected_weight) if target == "selected" else float(cfg.rejected_weight),
            configured_round=cfg.pseudo_round,
            source_template=source_template,
        )
        add_source_ref(group, queue_item=queue_item, ref=ref)

        for prediction in predictions_from_queue_item(queue_item):
            annotation = prediction_to_annotation(
                prediction,
                review_id=review_id,
                raw_prediction_ref=ref,
                width=width,
                height=height,
                clip_boxes=bool(cfg.clip_boxes),
                include_review_metadata=bool(cfg.include_review_metadata),
            )
            if annotation is None:
                dropped_prediction_count += 1
                continue
            group.annotations.append(annotation)
            task = str(annotation.get("task", ""))
            label = str(annotation.get("label", ""))
            if target == "selected":
                selected_task_counts[task] += 1
                selected_label_counts[label] += 1
            else:
                rejected_task_counts[task] += 1
                rejected_label_counts[label] += 1

    selected_rows = [group_to_row(group) for group in selected_groups.values() if group.annotations]
    rejected_rows = [group_to_row(group) for group in rejected_groups.values() if group.annotations]
    extra_decisions = len([review_id for review_id in decisions if review_id not in queue_review_ids])
    selected_annotations = sum(len(row["annotations"]) for row in selected_rows)
    rejected_annotations = sum(len(row["annotations"]) for row in rejected_rows)
    manifest = {
        "created_at": now_iso(),
        "dry_run": bool(cfg.dry_run),
        "raw_predictions_file": str(raw_predictions_file),
        "review_queue_file": str(review_queue_file),
        "review_decisions_file": str(review_decisions_file),
        "output_dir": str(output_dir),
        "selected_images": len(selected_rows),
        "selected_annotations": selected_annotations,
        "rejected_images": len(rejected_rows),
        "rejected_annotations": rejected_annotations,
        "pending_count": counts["pending"],
        "unsure_count": counts["unsure"],
        "accepted_count": counts["accepted"],
        "rejected_count": counts["rejected"],
        "ignored_count": counts["ignored"],
        "extra_decisions": extra_decisions,
        "dropped_prediction_count": dropped_prediction_count,
        "task_counts": dict(selected_task_counts),
        "label_counts": dict(selected_label_counts),
        "selected_task_counts": dict(selected_task_counts),
        "selected_label_counts": dict(selected_label_counts),
        "rejected_task_counts": dict(rejected_task_counts),
        "rejected_label_counts": dict(rejected_label_counts),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if bool(cfg.dry_run):
        summary = {
            "status": "dry_run",
            "selected_images": len(selected_rows),
            "selected_annotations": selected_annotations,
            "rejected_images": len(rejected_rows),
            "rejected_annotations": rejected_annotations,
            "pending_count": counts["pending"],
            "unsure_count": counts["unsure"],
            "task_counts": dict(selected_task_counts),
            "label_counts": dict(selected_label_counts),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    selected_path = output_dir / str(cfg.selected_output_file)
    rejected_path = output_dir / str(cfg.rejected_output_file)
    manifest_path = output_dir / str(cfg.manifest_file)
    if not bool(cfg.overwrite):
        existing = [path for path in (selected_path, rejected_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist and overwrite=false: {existing}")

    write_jsonl(selected_path, selected_rows)
    write_jsonl(rejected_path, rejected_rows)
    write_json(manifest_path, manifest)

    result = {
        "status": "ok",
        "selected_annotations_file": str(selected_path),
        "rejected_annotations_file": str(rejected_path),
        "selection_manifest": str(manifest_path),
        "selected_images": len(selected_rows),
        "selected_annotations": selected_annotations,
        "rejected_images": len(rejected_rows),
        "rejected_annotations": rejected_annotations,
        "pending_count": counts["pending"],
        "unsure_count": counts["unsure"],
        "task_counts": dict(selected_task_counts),
        "label_counts": dict(selected_label_counts),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="apply_review_decisions",
)
def main(cfg: DictConfig) -> None:
    apply_review_decisions(cfg)


if __name__ == "__main__":
    main()
