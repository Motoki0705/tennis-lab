"""Dataset-level orchestration for homography annotation evaluation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.contracts import (
    CourtAnnotationDatasetSpec,
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.evaluation.homography_quality import (
    evaluate_homography_quality,
)
from src.tasks.court_detection.evaluation.image_evidence import (
    image_diversity_metrics,
    line_edge_support,
)


@dataclass(frozen=True)
class DatasetEvaluationResult:
    """Serializable result for one annotation dataset."""

    dataset: CourtAnnotationDatasetSpec
    criteria: HomographyEvaluationCriteria
    records: list[dict[str, Any]]
    accepted_annotations: list[dict[str, Any]]
    rejected_annotations: list[dict[str, Any]]
    summary: dict[str, Any]


def evaluate_annotation_datasets(
    datasets: list[CourtAnnotationDatasetSpec],
    *,
    criteria: HomographyEvaluationCriteria,
    workers: int,
    use_refined_keypoints: bool,
) -> list[DatasetEvaluationResult]:
    """Evaluate multiple datasets in one deterministic invocation."""
    if not datasets:
        raise ValueError("At least one dataset must be configured.")
    names = [dataset.name for dataset in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"Dataset names must be unique, got {names}.")
    return [
        evaluate_annotation_dataset(
            dataset,
            criteria=criteria,
            workers=workers,
            use_refined_keypoints=use_refined_keypoints,
        )
        for dataset in datasets
    ]


def evaluate_annotation_dataset(
    dataset: CourtAnnotationDatasetSpec,
    *,
    criteria: HomographyEvaluationCriteria,
    workers: int,
    use_refined_keypoints: bool,
) -> DatasetEvaluationResult:
    """Evaluate one ``data_train.json``-compatible annotation file."""
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}.")
    entries = _load_entries(dataset.annotation_json)
    identifiers = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    valid_identifiers = [value for value in identifiers if isinstance(value, str)]
    duplicates = sorted(
        identifier
        for identifier, count in Counter(valid_identifiers).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate annotation ids in {dataset.annotation_json}: {duplicates[:5]}."
        )

    def evaluate_indexed(
        index_and_entry: tuple[int, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        index, entry = index_and_entry
        record, accepted = _evaluate_entry(
            entry,
            dataset=dataset,
            criteria=criteria,
            use_refined_keypoints=use_refined_keypoints,
        )
        return index, record, accepted

    indexed_entries = list(enumerate(entries))
    if workers == 1:
        evaluated = [evaluate_indexed(item) for item in indexed_entries]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(evaluate_indexed, indexed_entries))
    evaluated.sort(key=lambda item: item[0])

    records = [item[1] for item in evaluated]
    accepted_annotations: list[dict[str, Any]] = []
    for _index, _record, accepted_annotation in evaluated:
        if accepted_annotation is not None:
            accepted_annotations.append(accepted_annotation)
    rejected_annotations = [
        cast(dict[str, Any], entries[index])
        for index, record, _accepted in evaluated
        if not record["accepted"] and isinstance(entries[index], dict)
    ]
    return DatasetEvaluationResult(
        dataset=dataset,
        criteria=criteria,
        records=records,
        accepted_annotations=accepted_annotations,
        rejected_annotations=rejected_annotations,
        summary=_summarize(records),
    )


def write_evaluation_results(
    results: list[DatasetEvaluationResult],
    *,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Write per-dataset manifests plus one aggregate summary."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty and overwrite=false: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {"datasets": {}}
    aggregate_records: list[dict[str, Any]] = []
    for result in results:
        dataset_dir = output_dir / result.dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_json(dataset_dir / "records.json", result.records)
        _write_json(
            dataset_dir / "accepted_annotations.json", result.accepted_annotations
        )
        _write_json(
            dataset_dir / "rejected_annotations.json", result.rejected_annotations
        )
        payload = {
            "dataset": {
                "name": result.dataset.name,
                "annotation_json": str(result.dataset.annotation_json),
                "image_dir": str(result.dataset.image_dir),
                "image_extensions": list(result.dataset.image_extensions),
            },
            "criteria": asdict(result.criteria),
            "summary": result.summary,
        }
        _write_json(dataset_dir / "summary.json", payload)
        aggregate["datasets"][result.dataset.name] = payload
        aggregate_records.extend(
            {
                **record,
                "id": f"{result.dataset.name}:{record['id']}",
            }
            for record in result.records
        )
    aggregate["summary"] = _summarize(aggregate_records)
    _write_json(output_dir / "summary.json", aggregate)
    return output_dir / "summary.json"


def _load_entries(path: Path) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {path}")
    with path.open(encoding="utf-8") as file:
        payload: Any = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Annotation JSON must contain a top-level list: {path}")
    return payload


def _evaluate_entry(
    entry: Any,
    *,
    dataset: CourtAnnotationDatasetSpec,
    criteria: HomographyEvaluationCriteria,
    use_refined_keypoints: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(entry, dict):
        return _schema_rejection(None, "annotation_not_object"), None
    image_id = entry.get("id")
    if not isinstance(image_id, str) or not image_id:
        return _schema_rejection(None, "invalid_id"), None
    if Path(image_id).name != image_id:
        return _schema_rejection(image_id, "id_contains_path_components"), None
    try:
        keypoints = np.asarray(entry.get("kps"), dtype=np.float32)
    except (TypeError, ValueError):
        return _schema_rejection(image_id, "invalid_keypoints"), None
    if keypoints.shape != (14, 2) or not np.isfinite(keypoints).all():
        return _schema_rejection(image_id, "invalid_keypoints"), None

    image_path, resolution_error = _resolve_image_path(dataset, image_id)
    if image_path is None:
        return _schema_rejection(image_id, cast(str, resolution_error)), None
    image_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_raw is None:
        return _schema_rejection(
            image_id, "unreadable_image", image_path=image_path
        ), None
    image: NDArray[np.uint8] = np.asarray(image_raw, dtype=np.uint8)
    height, width = image.shape[:2]
    quality = evaluate_homography_quality(
        keypoints,
        image_width=width,
        image_height=height,
        criteria=criteria,
    )
    reasons = list(quality.rejection_reasons)
    support: float | None = None
    diversity: dict[str, float | str] = {}
    projected = quality.projected_keypoints_normalized
    if projected is not None:
        gray: NDArray[np.uint8] = np.asarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8
        )
        support = line_edge_support(
            gray,
            projected,
            distance_tolerance_px=criteria.line_distance_tolerance_px,
            max_side=criteria.line_evidence_max_side,
        )
        if support < criteria.min_line_edge_support:
            reasons.append("weak_line_evidence")
        diversity = image_diversity_metrics(image, projected)
    accepted = not reasons
    record = {
        "id": image_id,
        "image_path": str(image_path),
        "accepted": accepted,
        "rejection_reasons": reasons,
        "homography": _array_list(quality.homography),
        "projected_keypoints_normalized": _array_list(projected),
        "projected_keypoints_xy": (
            (projected * np.asarray([width - 1, height - 1], dtype=np.float32))
            .astype(float)
            .tolist()
            if projected is not None
            else None
        ),
        "inlier_mask": quality.inlier_mask.astype(int).tolist(),
        "residuals_normalized": quality.residuals_normalized.astype(float).tolist(),
        "metrics": {
            **quality.metrics,
            "line_edge_support": support,
            "image_width": width,
            "image_height": height,
        },
        "diversity": diversity,
    }
    if not accepted:
        return record, None
    accepted_entry = dict(entry)
    if use_refined_keypoints:
        projected_xy = cast(NDArray[np.float32], projected) * np.asarray(
            [width - 1, height - 1],
            dtype=np.float32,
        )
        accepted_entry["kps"] = projected_xy.astype(float).tolist()
    return record, accepted_entry


def _resolve_image_path(
    dataset: CourtAnnotationDatasetSpec,
    image_id: str,
) -> tuple[Path | None, str | None]:
    matches = [
        dataset.image_dir / f"{image_id}{extension}"
        for extension in dataset.image_extensions
        if (dataset.image_dir / f"{image_id}{extension}").is_file()
    ]
    if not matches:
        return None, "missing_image"
    if len(matches) > 1:
        return None, "ambiguous_image_extension"
    return matches[0], None


def _schema_rejection(
    image_id: str | None,
    reason: str,
    *,
    image_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "id": image_id,
        "image_path": str(image_path) if image_path is not None else None,
        "accepted": False,
        "rejection_reasons": [reason],
        "homography": None,
        "projected_keypoints_normalized": None,
        "projected_keypoints_xy": None,
        "inlier_mask": [],
        "residuals_normalized": [],
        "metrics": {},
        "diversity": {},
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [record for record in records if record["accepted"]]
    reasons = Counter(
        reason for record in records for reason in record["rejection_reasons"]
    )
    surface = Counter(
        record["diversity"].get("surface_color_bucket")
        for record in accepted
        if record["diversity"]
    )
    brightness = Counter(
        record["diversity"].get("brightness_bucket")
        for record in accepted
        if record["diversity"]
    )
    viewpoints = Counter(_viewpoint_bucket(record) for record in accepted)
    occupancy = Counter(_occupancy_bucket(record) for record in accepted)
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in accepted:
        hash_value = record["diversity"].get("dhash64")
        if isinstance(hash_value, str):
            hashes[hash_value].append(str(record["id"]))
    exact_duplicates = sorted(
        (identifiers for identifiers in hashes.values() if len(identifiers) > 1),
        key=lambda identifiers: (-len(identifiers), identifiers[0]),
    )
    return {
        "input_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(records) - len(accepted),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "rejection_reasons": dict(reasons.most_common()),
        "diversity": {
            "viewpoint_counts": dict(viewpoints),
            "court_occupancy_counts": dict(occupancy),
            "surface_color_counts": dict(surface),
            "brightness_counts": dict(brightness),
            "exact_dhash_duplicate_groups": exact_duplicates,
        },
    }


def _viewpoint_bucket(record: dict[str, Any]) -> str:
    ratio = float(record["metrics"]["opposite_edge_ratio"])
    if ratio < 0.6:
        return "perspective"
    if ratio < 0.9:
        return "oblique"
    return "weak_perspective"


def _occupancy_bucket(record: dict[str, Any]) -> str:
    area = float(record["metrics"]["court_area_ratio"])
    if area < 0.03:
        return "1.5_to_3_percent"
    if area < 0.10:
        return "3_to_10_percent"
    if area < 0.30:
        return "10_to_30_percent"
    return "at_least_30_percent"


def _array_list(array: np.ndarray | None) -> list[Any] | None:
    return array.astype(float).tolist() if array is not None else None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
