"""Overview:
Review Grounding DINO pseudo-label queue items with a fast binary OpenCV UI.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/review_binary.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/review_binary.py headless_auto_decision=accept max_items=2

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/review_binary.yaml`.
    - The same UI is used for CourtKP20 image-level reviews and tennis-role annotation-level reviews.
    - Decisions are compacted by `review_id` on every save so interrupted sessions can resume cleanly.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


DECISION_VALUES = {"accepted", "rejected", "unsure"}
AUTO_DECISIONS = {"none", "accept", "accepted", "reject", "rejected", "unsure"}

RIGHT_KEYS = {ord("j"), 83, 2555904, 65363}
LEFT_KEYS = {ord("k"), 81, 2424832, 65361}
ACCEPT_KEYS = {ord("a"), 10, 13}
REJECT_KEYS = {ord("r"), 8, 127}
UNSURE_KEYS = {ord("?"), ord("m")}
SAVE_KEYS = {ord("s")}
UNDO_KEYS = {ord("u")}
QUIT_KEYS = {ord("q"), 27}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(to_absolute_path(text)).resolve()


def normalize_auto_decision(value: Any) -> str:
    text = str(value or "none").strip().lower()
    if text not in AUTO_DECISIONS:
        raise ValueError(f"headless_auto_decision must be one of {sorted(AUTO_DECISIONS)}")
    if text == "accept":
        return "accepted"
    if text == "reject":
        return "rejected"
    return text


def normalize_decision(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"accept", "accepted"}:
        return "accepted"
    if text in {"reject", "rejected"}:
        return "rejected"
    if text == "unsure":
        return "unsure"
    return None


def resolve_path_value(value: Any, *, base_dir: Path, search_roots: list[Path]) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    raw = Path(text).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                base_dir / raw,
                base_dir.parent / raw,
                base_dir / "review_assets" / raw,
                Path(to_absolute_path(text)),
            ]
        )
        for root in search_roots:
            candidates.extend([root / raw, root / raw.name])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def load_queue(path: Path, max_items: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        review_id = str(row.get("review_id") or f"review_{index:06d}")
        if review_id in seen:
            raise ValueError(f"Duplicate review_id in queue: {review_id}")
        seen.add(review_id)
        normalized = dict(row)
        normalized["review_id"] = review_id
        queue.append(normalized)
    if max_items > 0:
        return queue[:max_items]
    return queue


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


def prediction_summary(item: dict[str, Any], max_parts: int = 3) -> str:
    predictions = item.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return "no predictions"

    parts: list[str] = []
    for prediction in predictions[:max_parts]:
        if not isinstance(prediction, dict):
            continue
        task = str(prediction.get("task") or "").strip()
        label = str(prediction.get("label") or "").strip()
        try:
            score = f"{float(prediction.get('score', 0.0)):.2f}"
        except (TypeError, ValueError):
            score = "0.00"
        name = "/".join(value for value in [task, label] if value)
        parts.append(f"{name or 'prediction'} {score}")
    if len(predictions) > max_parts:
        parts.append(f"+{len(predictions) - max_parts}")
    return " | ".join(parts) if parts else "predictions"


def make_decision_row(
    *,
    item: dict[str, Any],
    decision: str,
    operator_name: str | None,
    source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "review_id": str(item["review_id"]),
        "decision": decision,
        "decided_at": now_iso(),
        "review_unit": item.get("review_unit"),
        "image": item.get("image"),
        "source": source,
    }
    overlay_image = item.get("overlay_image")
    if overlay_image is not None:
        row["overlay_image"] = overlay_image
    if operator_name:
        row["operator"] = operator_name
    return row


def ordered_decision_rows(
    *,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    review_ids = [str(item["review_id"]) for item in queue]
    rows = [decisions[review_id] for review_id in review_ids if review_id in decisions]
    extras = [
        row
        for review_id, row in sorted(decisions.items())
        if review_id not in set(review_ids)
    ]
    return rows + extras


def build_summary(
    *,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    review_queue_file: Path,
    output_decisions_file: Path,
    dry_run: bool,
) -> dict[str, Any]:
    review_ids = {str(item["review_id"]) for item in queue}
    current_decisions = [
        row
        for review_id, row in decisions.items()
        if review_id in review_ids and normalize_decision(row.get("decision")) in DECISION_VALUES
    ]
    counts = Counter(str(row["decision"]) for row in current_decisions)
    decided = sum(counts.values())
    return {
        "updated_at": now_iso(),
        "dry_run": dry_run,
        "review_queue_file": str(review_queue_file),
        "output_decisions_file": str(output_decisions_file),
        "total": len(queue),
        "decided": decided,
        "accepted": counts.get("accepted", 0),
        "rejected": counts.get("rejected", 0),
        "unsure": counts.get("unsure", 0),
        "pending": max(0, len(queue) - decided),
        "extra_decisions": len([review_id for review_id in decisions if review_id not in review_ids]),
    }


def save_state(
    *,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    review_queue_file: Path,
    output_decisions_file: Path,
    summary_file: Path,
    dry_run: bool,
) -> dict[str, Any]:
    rows = ordered_decision_rows(queue=queue, decisions=decisions)
    write_jsonl_atomic(output_decisions_file, rows)
    summary = build_summary(
        queue=queue,
        decisions=decisions,
        review_queue_file=review_queue_file,
        output_decisions_file=output_decisions_file,
        dry_run=dry_run,
    )
    write_json(summary_file, summary)
    return summary


def find_next_index(
    *,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    current_index: int,
    direction: int,
    skip_decided: bool,
) -> int:
    if not queue:
        return 0
    if not skip_decided:
        return (current_index + direction) % len(queue)

    for offset in range(1, len(queue) + 1):
        candidate = (current_index + direction * offset) % len(queue)
        review_id = str(queue[candidate]["review_id"])
        if review_id not in decisions:
            return candidate
    return current_index


def initial_index(queue: list[dict[str, Any]], decisions: dict[str, dict[str, Any]], skip_decided: bool) -> int:
    if not queue or not skip_decided:
        return 0
    for index, item in enumerate(queue):
        if str(item["review_id"]) not in decisions:
            return index
    return 0


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for interactive review. Install opencv-python.") from exc
    return cv2


def import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for interactive review.") from exc
    return np


def read_cv_image(path: Path) -> Any | None:
    cv2 = import_cv2()
    np = import_numpy()
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None
    return image


def placeholder_image(width: int, height: int, text: str) -> Any:
    cv2 = import_cv2()
    np = import_numpy()
    image = np.full((height, width, 3), 56, dtype=np.uint8)
    cv2.putText(image, text, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    return image


def load_display_image(
    *,
    item: dict[str, Any],
    queue_dir: Path,
    image_search_roots: list[Path],
    placeholder_width: int,
    placeholder_height: int,
) -> Any:
    overlay_path = resolve_path_value(
        item.get("overlay_image"),
        base_dir=queue_dir,
        search_roots=image_search_roots,
    )
    if overlay_path is not None:
        image = read_cv_image(overlay_path)
        if image is not None:
            return image

    image_path = resolve_path_value(
        item.get("image"),
        base_dir=queue_dir,
        search_roots=image_search_roots,
    )
    if image_path is not None:
        image = read_cv_image(image_path)
        if image is not None:
            return image

    return placeholder_image(placeholder_width, placeholder_height, "Missing overlay/image")


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def render_display(
    *,
    item: dict[str, Any],
    index: int,
    total: int,
    decision: str | None,
    image: Any,
    max_display_width: int,
    max_display_height: int,
) -> Any:
    cv2 = import_cv2()
    np = import_numpy()
    header_height = 122
    image_height, image_width = image.shape[:2]
    scale = min(
        max_display_width / max(1, image_width),
        (max_display_height - header_height) / max(1, image_height),
        1.0,
    )
    if scale < 1.0:
        image = cv2.resize(image, (int(image_width * scale), int(image_height * scale)), interpolation=cv2.INTER_AREA)

    header = np.full((header_height, image.shape[1], 3), 28, dtype=np.uint8)
    canvas = np.vstack([header, image])

    status = decision or "pending"
    color = {
        "accepted": (80, 220, 110),
        "rejected": (80, 120, 255),
        "unsure": (70, 210, 245),
        "pending": (225, 225, 225),
    }.get(status, (225, 225, 225))
    title = f"{index + 1}/{total}  {item['review_id']}  [{status}]"
    summary = prediction_summary(item)
    unit = str(item.get("review_unit") or "")
    image_value = str(item.get("image") or "")
    help_text = "a/Enter accept | r/Backspace reject | ?/m unsure | j/right next | k/left prev | u undo | s save | q/Esc quit"

    cv2.putText(canvas, truncate_text(title, 110), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    cv2.putText(canvas, truncate_text(f"{unit}: {summary}", 120), (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    cv2.putText(canvas, truncate_text(image_value, 125), (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (195, 195, 195), 1)
    cv2.putText(canvas, truncate_text(help_text, 130), (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (165, 210, 255), 1)
    return canvas


def run_headless_auto(
    *,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    decision: str,
    max_items: int,
    operator_name: str | None,
) -> int:
    pending_items = [item for item in queue if str(item["review_id"]) not in decisions]
    limit = max_items if max_items > 0 else len(pending_items)
    for item in pending_items[:limit]:
        decisions[str(item["review_id"])] = make_decision_row(
            item=item,
            decision=decision,
            operator_name=operator_name,
            source="headless_auto_decision",
        )
    return min(limit, len(pending_items))


def run_interactive(
    *,
    cfg: DictConfig,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    review_queue_file: Path,
    output_decisions_file: Path,
    summary_file: Path,
    image_search_roots: list[Path],
) -> dict[str, Any]:
    cv2 = import_cv2()
    queue_dir = review_queue_file.parent
    skip_decided = not bool(cfg.show_decided)
    index = initial_index(queue, decisions, skip_decided)
    history: list[str] = []
    operator_name = str(cfg.operator).strip() if cfg.operator is not None and str(cfg.operator).strip() else None
    window_name = str(cfg.window_name)

    while queue:
        item = queue[index]
        review_id = str(item["review_id"])
        decision = normalize_decision(decisions.get(review_id, {}).get("decision"))
        image = load_display_image(
            item=item,
            queue_dir=queue_dir,
            image_search_roots=image_search_roots,
            placeholder_width=int(cfg.placeholder_width),
            placeholder_height=int(cfg.placeholder_height),
        )
        canvas = render_display(
            item=item,
            index=index,
            total=len(queue),
            decision=decision,
            image=image,
            max_display_width=int(cfg.max_display_width),
            max_display_height=int(cfg.max_display_height),
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKeyEx(0)

        if key in QUIT_KEYS:
            break
        if key in SAVE_KEYS:
            save_state(
                queue=queue,
                decisions=decisions,
                review_queue_file=review_queue_file,
                output_decisions_file=output_decisions_file,
                summary_file=summary_file,
                dry_run=False,
            )
            continue
        if key in RIGHT_KEYS:
            index = find_next_index(
                queue=queue,
                decisions=decisions,
                current_index=index,
                direction=1,
                skip_decided=skip_decided,
            )
            continue
        if key in LEFT_KEYS:
            index = find_next_index(
                queue=queue,
                decisions=decisions,
                current_index=index,
                direction=-1,
                skip_decided=skip_decided,
            )
            continue
        if key in UNDO_KEYS:
            undo_review_id = history.pop() if history else review_id
            decisions.pop(undo_review_id, None)
            for candidate_index, candidate in enumerate(queue):
                if str(candidate["review_id"]) == undo_review_id:
                    index = candidate_index
                    break
            save_state(
                queue=queue,
                decisions=decisions,
                review_queue_file=review_queue_file,
                output_decisions_file=output_decisions_file,
                summary_file=summary_file,
                dry_run=False,
            )
            continue

        new_decision: str | None = None
        if key in ACCEPT_KEYS:
            new_decision = "accepted"
        elif key in REJECT_KEYS:
            new_decision = "rejected"
        elif key in UNSURE_KEYS:
            new_decision = "unsure"

        if new_decision is not None:
            decisions[review_id] = make_decision_row(
                item=item,
                decision=new_decision,
                operator_name=operator_name,
                source="interactive",
            )
            history.append(review_id)
            save_state(
                queue=queue,
                decisions=decisions,
                review_queue_file=review_queue_file,
                output_decisions_file=output_decisions_file,
                summary_file=summary_file,
                dry_run=False,
            )
            if bool(cfg.auto_advance_after_decision):
                index = find_next_index(
                    queue=queue,
                    decisions=decisions,
                    current_index=index,
                    direction=1,
                    skip_decided=skip_decided,
                )

    cv2.destroyAllWindows()
    return save_state(
        queue=queue,
        decisions=decisions,
        review_queue_file=review_queue_file,
        output_decisions_file=output_decisions_file,
        summary_file=summary_file,
        dry_run=False,
    )


def run_review_binary(cfg: DictConfig) -> dict[str, Any]:
    review_queue_file = Path(to_absolute_path(str(cfg.review_queue_file))).resolve()
    if not review_queue_file.is_file():
        raise FileNotFoundError(f"review_queue_file not found: {review_queue_file}")

    output_decisions_file = Path(to_absolute_path(str(cfg.output_decisions_file))).resolve()
    summary_file = optional_path(cfg.summary_file) or output_decisions_file.with_name("review_summary.json")
    image_search_roots = [
        Path(to_absolute_path(str(value))).resolve()
        for value in cfg.image_search_roots
    ]

    queue = load_queue(review_queue_file, max_items=int(cfg.max_items))
    decisions = load_latest_decisions(output_decisions_file)
    auto_decision = normalize_auto_decision(cfg.headless_auto_decision)

    if bool(cfg.dry_run):
        summary = build_summary(
            queue=queue,
            decisions=decisions,
            review_queue_file=review_queue_file,
            output_decisions_file=output_decisions_file,
            dry_run=True,
        )
        summary["status"] = "dry_run"
        summary["config"] = OmegaConf.to_container(cfg, resolve=True)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    if auto_decision != "none":
        operator_name = str(cfg.operator).strip() if cfg.operator is not None and str(cfg.operator).strip() else None
        auto_count = run_headless_auto(
            queue=queue,
            decisions=decisions,
            decision=auto_decision,
            max_items=int(cfg.max_items),
            operator_name=operator_name,
        )
        summary = save_state(
            queue=queue,
            decisions=decisions,
            review_queue_file=review_queue_file,
            output_decisions_file=output_decisions_file,
            summary_file=summary_file,
            dry_run=False,
        )
        summary["status"] = "ok"
        summary["headless_auto_decision"] = auto_decision
        summary["auto_decided"] = auto_count
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    if not queue:
        summary = save_state(
            queue=queue,
            decisions=decisions,
            review_queue_file=review_queue_file,
            output_decisions_file=output_decisions_file,
            summary_file=summary_file,
            dry_run=False,
        )
        summary["status"] = "empty_queue"
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return summary

    summary = run_interactive(
        cfg=cfg,
        queue=queue,
        decisions=decisions,
        review_queue_file=review_queue_file,
        output_decisions_file=output_decisions_file,
        summary_file=summary_file,
        image_search_roots=image_search_roots,
    )
    summary["status"] = "ok"
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="review_binary",
)
def main(cfg: DictConfig) -> None:
    run_review_binary(cfg)


if __name__ == "__main__":
    main()
