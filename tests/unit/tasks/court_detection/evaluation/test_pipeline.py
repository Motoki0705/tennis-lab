"""Tests for multi-dataset court annotation evaluation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation import (
    CourtAnnotationDatasetSpec,
    HomographyEvaluationCriteria,
    evaluate_annotation_dataset,
    evaluate_annotation_datasets,
    write_evaluation_results,
)
from src.tasks.court_detection.evaluation.image_evidence import COURT_LINE_EDGES
from src.tasks.court_detection.geometry import court_template_xy, project_points


def _write_valid_dataset(root: Path, *, name: str) -> CourtAnnotationDatasetSpec:
    image_dir = root / name / "images"
    image_dir.mkdir(parents=True)
    width, height = 1280, 720
    homography = np.asarray(
        [
            [0.040, 0.002, 0.50],
            [0.002, -0.025, 0.48],
            [0.002, -0.012, 1.00],
        ],
        dtype=np.float32,
    )
    projected = project_points(court_template_xy(), homography)
    pixels = projected * np.asarray([width - 1, height - 1], dtype=np.float32)
    image: NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)
    for first, second in COURT_LINE_EDGES:
        cv2.line(
            image,
            tuple(np.rint(pixels[first]).astype(int)),
            tuple(np.rint(pixels[second]).astype(int)),
            (255, 255, 255),
            4,
        )
    assert cv2.imwrite(str(image_dir / "valid.jpg"), image)
    annotation_json = root / name / "data_train.json"
    annotation_json.write_text(
        json.dumps(
            [
                {"id": "valid", "metric": 1.0, "kps": pixels.tolist()},
                {"id": "missing", "metric": 1.0, "kps": pixels.tolist()},
            ]
        ),
        encoding="utf-8",
    )
    return CourtAnnotationDatasetSpec(
        name=name,
        annotation_json=annotation_json,
        image_dir=image_dir,
    )


def _criteria() -> HomographyEvaluationCriteria:
    return HomographyEvaluationCriteria(
        min_court_area_ratio=0.001,
        min_line_edge_support=0.8,
    )


def test_pipeline_reports_missing_images_and_writes_refined_annotations(
    tmp_path: Path,
) -> None:
    dataset = _write_valid_dataset(tmp_path, name="first")

    result = evaluate_annotation_dataset(
        dataset,
        criteria=_criteria(),
        workers=2,
        use_refined_keypoints=True,
    )

    assert result.summary["accepted_count"] == 1
    assert result.summary["rejected_count"] == 1
    assert result.records[1]["rejection_reasons"] == ["missing_image"]
    assert result.accepted_annotations[0]["id"] == "valid"
    assert len(result.accepted_annotations[0]["kps"]) == 14


def test_pipeline_evaluates_multiple_named_datasets_and_writes_aggregate(
    tmp_path: Path,
) -> None:
    first = _write_valid_dataset(tmp_path, name="first")
    second = _write_valid_dataset(tmp_path, name="second")

    results = evaluate_annotation_datasets(
        [first, second],
        criteria=_criteria(),
        workers=1,
    )
    summary_path = write_evaluation_results(
        results,
        output_dir=tmp_path / "outputs",
        overwrite=False,
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload["datasets"]) == {"first", "second"}
    assert payload["summary"]["accepted_count"] == 2
    assert payload["summary"]["rejected_count"] == 2
    assert (summary_path.parent / "first" / "accepted_annotations.json").is_file()


def test_pipeline_rejects_ambiguous_image_extensions(tmp_path: Path) -> None:
    dataset = _write_valid_dataset(tmp_path, name="ambiguous")
    source = dataset.image_dir / "valid.jpg"
    duplicate = dataset.image_dir / "valid.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    assert image is not None
    assert cv2.imwrite(str(duplicate), image)

    result = evaluate_annotation_dataset(dataset, criteria=_criteria())

    assert result.records[0]["rejection_reasons"] == ["ambiguous_image_extension"]
