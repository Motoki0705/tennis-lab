"""Evaluate one or more court-keypoint JSON datasets through homography fitting.

Usage:
    python -m src.tasks.court_detection.scripts.evaluate_homography_annotations
    python -m src.tasks.court_detection.scripts.evaluate_homography_annotations evaluate_homography_annotations.datasets.0.annotation_json=court/data_val.json

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/evaluate_homography_annotations.yaml`.
    - Every input JSON must follow the `data/court/data_train.json` list format with `id` and 14-point `kps` fields.
    - Rejected files are reported but never deleted by this command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.court_detection.configuration import validate_paths_boundary
from src.tasks.court_detection.evaluation import (
    CourtAnnotationDatasetSpec,
    HomographyEvaluationCriteria,
    evaluate_annotation_datasets,
    write_evaluation_results,
)
from src.utils.configuration import PathResolver, PathRole
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.evaluate_homography_annotations"


@hydra_main(
    config_path="../configs",
    config_name="evaluate_homography_annotations",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    datasets, criteria, workers, use_refined_keypoints, output_dir, overwrite = (
        _runtime(cfg)
    )
    results = evaluate_annotation_datasets(
        datasets,
        criteria=criteria,
        workers=workers,
        use_refined_keypoints=use_refined_keypoints,
    )
    summary_path = write_evaluation_results(
        results,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    for result in results:
        print(
            f"{result.dataset.name}: accepted={result.summary['accepted_count']} "
            f"rejected={result.summary['rejected_count']}"
        )
    print(f"summary={summary_path}")
    return 0


def _runtime(
    cfg: DictConfig,
) -> tuple[
    list[CourtAnnotationDatasetSpec],
    HomographyEvaluationCriteria,
    int,
    bool,
    Path,
    bool,
]:
    root, resolver = validate_paths_boundary(
        cfg, expected_sections={"evaluate_homography_annotations"}
    )
    section = require_config_mapping(
        root, "evaluate_homography_annotations", path="configuration"
    )
    expected = {
        "datasets",
        "output_dir",
        "workers",
        "overwrite",
        "use_refined_keypoints",
        "criteria",
    }
    if set(section) != expected:
        raise ValueError(
            f"evaluate_homography_annotations requires exactly {sorted(expected)}."
        )
    datasets_raw = require_config_value(
        section, "datasets", list, path="evaluate_homography_annotations"
    )
    datasets = [
        _dataset_from_mapping(item, resolver=resolver)
        for item in cast("list[object]", datasets_raw)
    ]
    if not datasets:
        raise ValueError("evaluate_homography_annotations.datasets must not be empty.")
    criteria_raw = require_config_mapping(
        section, "criteria", path="evaluate_homography_annotations"
    )
    criteria_keys = {
        "ransac_reproj_threshold_normalized",
        "min_inliers",
        "min_template_x_span_ratio",
        "min_template_y_span_ratio",
        "max_inlier_rms_normalized",
        "min_visible_fraction",
        "min_court_area_ratio",
        "max_court_area_ratio",
        "min_line_edge_support",
        "line_distance_tolerance_px",
        "line_evidence_max_side",
        "require_ground_view",
        "max_opposite_edge_ratio",
    }
    if set(criteria_raw) != criteria_keys:
        raise ValueError(
            f"evaluation criteria requires exactly {sorted(criteria_keys)}."
        )
    for key in criteria_keys - {
        "min_inliers",
        "line_evidence_max_side",
        "require_ground_view",
    }:
        require_config_value(
            criteria_raw,
            key,
            (float, int),
            path="evaluate_homography_annotations.criteria",
        )
    for key in ("min_inliers", "line_evidence_max_side"):
        require_config_value(
            criteria_raw, key, int, path="evaluate_homography_annotations.criteria"
        )
    require_config_value(
        criteria_raw,
        "require_ground_view",
        bool,
        path="evaluate_homography_annotations.criteria",
    )
    criteria = HomographyEvaluationCriteria(
        **cast("dict[str, Any]", dict(criteria_raw))
    )
    workers = cast(
        "int",
        require_config_value(
            section, "workers", int, path="evaluate_homography_annotations"
        ),
    )
    if workers <= 0:
        raise ValueError("evaluate_homography_annotations.workers must be positive.")
    return (
        datasets,
        criteria,
        workers,
        cast(
            "bool",
            require_config_value(
                section,
                "use_refined_keypoints",
                bool,
                path="evaluate_homography_annotations",
            ),
        ),
        resolver.resolve(
            PathRole.OUTPUT,
            str(
                require_config_value(
                    section, "output_dir", str, path="evaluate_homography_annotations"
                )
            ),
        ),
        cast(
            "bool",
            require_config_value(
                section, "overwrite", bool, path="evaluate_homography_annotations"
            ),
        ),
    )


def _validate_boundary(cfg: DictConfig) -> None:
    _runtime(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


def _dataset_from_mapping(
    item: Any, *, resolver: PathResolver
) -> CourtAnnotationDatasetSpec:
    if not isinstance(item, dict):
        raise TypeError(
            f"Each dataset config must be a mapping, got {type(item).__name__}."
        )
    if set(item) != {"name", "annotation_json", "image_dir", "image_extensions"}:
        raise ValueError(
            "Each dataset requires exactly name, annotation_json, image_dir, image_extensions."
        )
    extensions = item["image_extensions"]
    if any(
        type(item[key]) is not str for key in ("name", "annotation_json", "image_dir")
    ):
        raise TypeError("dataset name and paths must be strings.")
    if (
        not isinstance(extensions, list)
        or not extensions
        or not all(type(value) is str and bool(value) for value in extensions)
    ):
        raise TypeError("dataset.image_extensions must be a list of strings.")
    if not str(item["name"]):
        raise ValueError("dataset.name must not be empty.")
    return CourtAnnotationDatasetSpec(
        name=str(item["name"]),
        annotation_json=resolver.resolve(PathRole.DATA, str(item["annotation_json"])),
        image_dir=resolver.resolve(PathRole.DATA, str(item["image_dir"])),
        image_extensions=tuple(extensions),
    )


if __name__ == "__main__":
    cast(Any, main)()
