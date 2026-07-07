"""Typed manifest contracts for automated ball-detector evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import OmegaConf

from src.utils.paths import resolve_project_path

EvaluationCategory = Literal["architecture-controlled", "full-strategy"]
_CATEGORIES = {"architecture-controlled", "full-strategy"}
_ALLOWED_SPLITS = {"val", "test"}
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class MetricsSpec:
    """Detection thresholds shared by every comparison job."""

    peak_threshold: float = 0.5
    ball_distance_threshold: float = 4.0
    nms_kernel: int = 9
    max_predictions_per_frame: int = 8
    subpixel_refine: bool = True


@dataclass(frozen=True)
class PerformanceSpec:
    """Inference timing controls."""

    warmup_batches: int = 1
    max_batches_per_split: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """One fixed evaluation dataset and its allowed splits."""

    id: str
    config: str
    splits: tuple[str, ...]
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint and the datasets on which it must be evaluated."""

    id: str
    category: EvaluationCategory
    checkpoint: Path
    expected_model_name: str
    datasets: tuple[str, ...]
    enabled: bool = True
    strict: bool = True
    weights_only: bool = False


@dataclass(frozen=True)
class EvaluationManifest:
    """Top-level evaluation manifest."""

    schema: str
    output_dir: Path
    device: str
    resume: bool
    fail_fast: bool
    metrics: MetricsSpec
    performance: PerformanceSpec
    datasets: dict[str, DatasetSpec]
    models: tuple[ModelSpec, ...]


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    """Load and validate an evaluation manifest from YAML."""
    manifest_path = resolve_project_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    raw = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Evaluation manifest root must be a mapping.")

    schema = str(raw.get("schema", ""))
    if schema != "ball_detection_evaluation_manifest_v1":
        raise ValueError(
            "Unsupported evaluation manifest schema: "
            f"{schema!r}; expected 'ball_detection_evaluation_manifest_v1'."
        )
    datasets = _parse_datasets(raw.get("datasets"))
    models = _parse_models(raw.get("models"), datasets)
    metrics_raw = _mapping(raw.get("metrics", {}), name="metrics")
    performance_raw = _mapping(raw.get("performance", {}), name="performance")
    metrics = MetricsSpec(
        peak_threshold=float(metrics_raw.get("peak_threshold", 0.5)),
        ball_distance_threshold=float(
            metrics_raw.get("ball_distance_threshold", 4.0)
        ),
        nms_kernel=int(metrics_raw.get("nms_kernel", 9)),
        max_predictions_per_frame=int(
            metrics_raw.get("max_predictions_per_frame", 8)
        ),
        subpixel_refine=bool(metrics_raw.get("subpixel_refine", True)),
    )
    performance = PerformanceSpec(
        warmup_batches=int(performance_raw.get("warmup_batches", 1)),
        max_batches_per_split=_optional_int(
            performance_raw.get("max_batches_per_split")
        ),
    )
    _validate_thresholds(metrics, performance)

    output_dir = resolve_project_path(
        str(raw.get("output_dir", "outputs/ball_detection/evaluation"))
    )
    return EvaluationManifest(
        schema=schema,
        output_dir=output_dir,
        device=str(raw.get("device", "auto")),
        resume=bool(raw.get("resume", True)),
        fail_fast=bool(raw.get("fail_fast", False)),
        metrics=metrics,
        performance=performance,
        datasets=datasets,
        models=models,
    )


def _parse_datasets(value: object) -> dict[str, DatasetSpec]:
    raw_datasets = _mapping(value, name="datasets")
    if not raw_datasets:
        raise ValueError("Evaluation manifest must define at least one dataset.")
    parsed: dict[str, DatasetSpec] = {}
    for dataset_id, raw_spec in raw_datasets.items():
        _validate_identifier(str(dataset_id), name="dataset id")
        spec = _mapping(raw_spec, name=f"datasets.{dataset_id}")
        splits = tuple(str(split) for split in _list(spec.get("splits"), "splits"))
        if not splits:
            raise ValueError(f"datasets.{dataset_id}.splits must not be empty.")
        unsupported = set(splits).difference(_ALLOWED_SPLITS)
        if unsupported:
            raise ValueError(
                f"datasets.{dataset_id} contains forbidden splits "
                f"{sorted(unsupported)}; evaluation may only access val/test."
            )
        if len(set(splits)) != len(splits):
            raise ValueError(f"datasets.{dataset_id}.splits contains duplicates.")
        overrides = _mapping(
            spec.get("overrides", {}),
            name=f"datasets.{dataset_id}.overrides",
        )
        parsed[str(dataset_id)] = DatasetSpec(
            id=str(dataset_id),
            config=str(spec.get("config", "")).strip(),
            splits=splits,
            overrides=overrides,
        )
        if not parsed[str(dataset_id)].config:
            raise ValueError(f"datasets.{dataset_id}.config must not be empty.")
    return parsed


def _parse_models(
    value: object,
    datasets: dict[str, DatasetSpec],
) -> tuple[ModelSpec, ...]:
    raw_models = _list(value, "models")
    if not raw_models:
        raise ValueError("Evaluation manifest must define at least one model.")
    parsed: list[ModelSpec] = []
    seen_ids: set[str] = set()
    for index, raw_model in enumerate(raw_models):
        spec = _mapping(raw_model, name=f"models[{index}]")
        model_id = str(spec.get("id", "")).strip()
        if not model_id:
            raise ValueError(f"models[{index}].id must not be empty.")
        _validate_identifier(model_id, name=f"models[{index}].id")
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model id: {model_id!r}.")
        seen_ids.add(model_id)
        category = str(spec.get("category", ""))
        if category not in _CATEGORIES:
            raise ValueError(
                f"models[{index}].category must be one of {sorted(_CATEGORIES)}, "
                f"got {category!r}."
            )
        dataset_ids = tuple(
            str(dataset_id)
            for dataset_id in _list(spec.get("datasets"), "datasets")
        )
        if not dataset_ids:
            raise ValueError(f"models[{index}].datasets must not be empty.")
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError(f"models[{index}].datasets contains duplicates.")
        missing = set(dataset_ids).difference(datasets)
        if missing:
            raise ValueError(
                f"models[{index}] references unknown datasets {sorted(missing)}."
            )
        checkpoint_value = str(spec.get("checkpoint", "")).strip()
        if not checkpoint_value:
            raise ValueError(f"models[{index}].checkpoint must not be empty.")
        checkpoint = resolve_project_path(checkpoint_value)
        parsed.append(
            ModelSpec(
                id=model_id,
                category=cast(EvaluationCategory, category),
                checkpoint=checkpoint,
                expected_model_name=str(
                    spec.get("expected_model_name", "")
                ).strip(),
                datasets=dataset_ids,
                enabled=bool(spec.get("enabled", True)),
                strict=bool(spec.get("strict", True)),
                weights_only=bool(spec.get("weights_only", False)),
            )
        )
        if not parsed[-1].expected_model_name:
            raise ValueError(
                f"models[{index}].expected_model_name must not be empty."
            )
    return tuple(parsed)


def _validate_thresholds(
    metrics: MetricsSpec,
    performance: PerformanceSpec,
) -> None:
    if metrics.peak_threshold < 0:
        raise ValueError("metrics.peak_threshold must be non-negative.")
    if metrics.ball_distance_threshold < 0:
        raise ValueError("metrics.ball_distance_threshold must be non-negative.")
    if metrics.nms_kernel <= 0 or metrics.nms_kernel % 2 == 0:
        raise ValueError("metrics.nms_kernel must be a positive odd integer.")
    if metrics.max_predictions_per_frame <= 0:
        raise ValueError("metrics.max_predictions_per_frame must be positive.")
    if performance.warmup_batches < 0:
        raise ValueError("performance.warmup_batches must be non-negative.")
    if (
        performance.max_batches_per_split is not None
        and performance.max_batches_per_split <= 0
    ):
        raise ValueError(
            "performance.max_batches_per_split must be positive when set."
        )


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, str)):
        raise TypeError("Expected an integer or null.")
    return int(value)


def _validate_identifier(value: str, *, name: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must contain only letters, digits, '.', '_', and '-', "
            f"got {value!r}."
        )


__all__ = [
    "DatasetSpec",
    "EvaluationCategory",
    "EvaluationManifest",
    "MetricsSpec",
    "ModelSpec",
    "PerformanceSpec",
    "load_evaluation_manifest",
]
