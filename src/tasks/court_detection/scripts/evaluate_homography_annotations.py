"""Evaluate one or more court-keypoint JSON datasets through homography fitting.

Usage:
    python -m src.tasks.court_detection.scripts.evaluate_homography_annotations
    python -m src.tasks.court_detection.scripts.evaluate_homography_annotations evaluate_homography_annotations.datasets.0.annotation_json=data/court/data_val.json

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/evaluate_homography_annotations.yaml`.
    - Every input JSON must follow the `data/court/data_train.json` list format with `id` and 14-point `kps` fields.
    - Rejected files are reported but never deleted by this command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.evaluation import (
    CourtAnnotationDatasetSpec,
    HomographyEvaluationCriteria,
    evaluate_annotation_datasets,
    write_evaluation_results,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="evaluate_homography_annotations",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    root = Path(get_original_cwd()).resolve()
    raw = OmegaConf.to_container(
        cfg.evaluate_homography_annotations,
        resolve=True,
    )
    if not isinstance(raw, dict):
        raise TypeError("evaluate_homography_annotations config must be a mapping.")
    datasets_raw = raw.get("datasets")
    if not isinstance(datasets_raw, list):
        raise TypeError("evaluate_homography_annotations.datasets must be a list.")
    datasets = [_dataset_from_mapping(item, root=root) for item in datasets_raw]
    criteria_raw = raw.get("criteria")
    if not isinstance(criteria_raw, dict):
        raise TypeError("evaluate_homography_annotations.criteria must be a mapping.")
    criteria = HomographyEvaluationCriteria(**cast(dict[str, Any], criteria_raw))
    results = evaluate_annotation_datasets(
        datasets,
        criteria=criteria,
        workers=int(raw.get("workers", 1)),
        use_refined_keypoints=bool(raw.get("use_refined_keypoints", True)),
    )
    output_dir = _resolve_path(root, str(raw["output_dir"]))
    summary_path = write_evaluation_results(
        results,
        output_dir=output_dir,
        overwrite=bool(raw.get("overwrite", False)),
    )
    for result in results:
        print(
            f"{result.dataset.name}: accepted={result.summary['accepted_count']} "
            f"rejected={result.summary['rejected_count']}"
        )
    print(f"summary={summary_path}")
    return 0


def _dataset_from_mapping(item: Any, *, root: Path) -> CourtAnnotationDatasetSpec:
    if not isinstance(item, dict):
        raise TypeError(
            f"Each dataset config must be a mapping, got {type(item).__name__}."
        )
    extensions = item.get("image_extensions", [".png", ".jpg", ".jpeg"])
    if not isinstance(extensions, list) or not all(
        isinstance(value, str) for value in extensions
    ):
        raise TypeError("dataset.image_extensions must be a list of strings.")
    return CourtAnnotationDatasetSpec(
        name=str(item["name"]),
        annotation_json=_resolve_path(root, str(item["annotation_json"])),
        image_dir=_resolve_path(root, str(item["image_dir"])),
        image_extensions=tuple(extensions),
    )


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    cast(Any, main)()
