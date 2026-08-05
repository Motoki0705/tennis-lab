"""Typed manifest contracts for automated ball-detector evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import OmegaConf

from src.tasks.ball_detection.configuration import exact_mapping
from src.utils.configuration import PathResolver, PathRole

EvaluationCategory = Literal["architecture-controlled", "full-strategy"]
_CATEGORIES = {"architecture-controlled", "full-strategy"}
_ALLOWED_SPLITS = {"val", "test"}
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class MetricsSpec:
    """Detection thresholds shared by every comparison job."""

    peak_threshold: float
    ball_distance_threshold: float
    nms_kernel: int
    max_predictions_per_frame: int
    subpixel_refine: bool


@dataclass(frozen=True)
class PerformanceSpec:
    """Inference timing controls."""

    warmup_batches: int
    max_batches_per_split: int | None


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
    enabled: bool
    strict: bool
    weights_only: bool


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
    resolver: PathResolver


def load_evaluation_manifest(
    path: str | Path, *, resolver: PathResolver
) -> EvaluationManifest:
    """Load and validate an evaluation manifest from YAML."""
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        raise ValueError("Evaluation manifest path must be absolute.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    raw = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Evaluation manifest root must be a mapping.")

    raw = dict(
        exact_mapping(
            raw,
            path="manifest",
            required={
                "schema",
                "output_dir",
                "device",
                "resume",
                "fail_fast",
                "metrics",
                "performance",
                "datasets",
                "models",
            },
        )
    )
    schema = _string(raw["schema"], name="manifest.schema")
    if schema != "ball_detection_evaluation_manifest_v1":
        raise ValueError(
            "Unsupported evaluation manifest schema: "
            f"{schema!r}; expected 'ball_detection_evaluation_manifest_v1'."
        )
    datasets = _parse_datasets(raw["datasets"])
    models = _parse_models(raw["models"], datasets, resolver=resolver)
    metrics_raw = dict(
        exact_mapping(
            raw["metrics"],
            path="manifest.metrics",
            required={
                "peak_threshold",
                "ball_distance_threshold",
                "nms_kernel",
                "max_predictions_per_frame",
                "subpixel_refine",
            },
        )
    )
    performance_raw = dict(
        exact_mapping(
            raw["performance"],
            path="manifest.performance",
            required={"warmup_batches", "max_batches_per_split"},
        )
    )
    metrics = MetricsSpec(
        peak_threshold=_number(
            metrics_raw["peak_threshold"], name="manifest.metrics.peak_threshold"
        ),
        ball_distance_threshold=_number(
            metrics_raw["ball_distance_threshold"],
            name="manifest.metrics.ball_distance_threshold",
        ),
        nms_kernel=_integer(
            metrics_raw["nms_kernel"], name="manifest.metrics.nms_kernel"
        ),
        max_predictions_per_frame=_integer(
            metrics_raw["max_predictions_per_frame"],
            name="manifest.metrics.max_predictions_per_frame",
        ),
        subpixel_refine=_boolean(
            metrics_raw["subpixel_refine"],
            name="manifest.metrics.subpixel_refine",
        ),
    )
    performance = PerformanceSpec(
        warmup_batches=_integer(
            performance_raw["warmup_batches"],
            name="manifest.performance.warmup_batches",
        ),
        max_batches_per_split=_optional_int(performance_raw["max_batches_per_split"]),
    )
    _validate_thresholds(metrics, performance)

    output_dir = resolver.resolve(
        PathRole.OUTPUT,
        _string(raw["output_dir"], name="manifest.output_dir"),
    )
    return EvaluationManifest(
        schema=schema,
        output_dir=output_dir,
        device=_string(raw["device"], name="manifest.device"),
        resume=_boolean(raw["resume"], name="manifest.resume"),
        fail_fast=_boolean(raw["fail_fast"], name="manifest.fail_fast"),
        metrics=metrics,
        performance=performance,
        datasets=datasets,
        models=models,
        resolver=resolver,
    )


def _parse_datasets(value: object) -> dict[str, DatasetSpec]:
    raw_datasets = _mapping(value, name="datasets")
    if not raw_datasets:
        raise ValueError("Evaluation manifest must define at least one dataset.")
    parsed: dict[str, DatasetSpec] = {}
    for dataset_id, raw_spec in raw_datasets.items():
        _validate_identifier(dataset_id, name="dataset id")
        spec = dict(
            exact_mapping(
                raw_spec,
                path=f"manifest.datasets.{dataset_id}",
                required={"config", "splits", "overrides"},
            )
        )
        splits = tuple(
            _string(split, name=f"datasets.{dataset_id}.splits item")
            for split in _list(spec["splits"], "splits")
        )
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
            spec["overrides"],
            name=f"datasets.{dataset_id}.overrides",
        )
        config_name = _string(
            spec["config"], name=f"datasets.{dataset_id}.config"
        )
        _validate_identifier(config_name, name=f"datasets.{dataset_id}.config")
        parsed[dataset_id] = DatasetSpec(
            id=dataset_id,
            config=config_name,
            splits=splits,
            overrides=overrides,
        )
    return parsed


def _parse_models(
    value: object,
    datasets: dict[str, DatasetSpec],
    *,
    resolver: PathResolver,
) -> tuple[ModelSpec, ...]:
    raw_models = _list(value, "models")
    if not raw_models:
        raise ValueError("Evaluation manifest must define at least one model.")
    parsed: list[ModelSpec] = []
    seen_ids: set[str] = set()
    for index, raw_model in enumerate(raw_models):
        spec = dict(
            exact_mapping(
                raw_model,
                path=f"manifest.models[{index}]",
                required={
                    "id",
                    "category",
                    "checkpoint",
                    "expected_model_name",
                    "datasets",
                    "enabled",
                    "strict",
                    "weights_only",
                },
            )
        )
        model_id = _string(spec["id"], name=f"models[{index}].id")
        _validate_identifier(model_id, name=f"models[{index}].id")
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model id: {model_id!r}.")
        seen_ids.add(model_id)
        category = _string(spec["category"], name=f"models[{index}].category")
        if category not in _CATEGORIES:
            raise ValueError(
                f"models[{index}].category must be one of {sorted(_CATEGORIES)}, "
                f"got {category!r}."
            )
        dataset_ids = tuple(
            _string(dataset_id, name=f"models[{index}].datasets item")
            for dataset_id in _list(spec["datasets"], "datasets")
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
        checkpoint_value = _string(
            spec["checkpoint"], name=f"models[{index}].checkpoint"
        )
        checkpoint = resolver.resolve(PathRole.CHECKPOINT, checkpoint_value)
        expected_model_name = _string(
            spec["expected_model_name"],
            name=f"models[{index}].expected_model_name",
        )
        parsed.append(
            ModelSpec(
                id=model_id,
                category=cast(EvaluationCategory, category),
                checkpoint=checkpoint,
                expected_model_name=expected_model_name,
                datasets=dataset_ids,
                enabled=_boolean(spec["enabled"], name=f"models[{index}].enabled"),
                strict=_boolean(spec["strict"], name=f"models[{index}].strict"),
                weights_only=_boolean(
                    spec["weights_only"], name=f"models[{index}].weights_only"
                ),
            )
        )
    if not any(model.enabled for model in parsed):
        raise ValueError("Evaluation manifest must enable at least one model.")
    return tuple(parsed)


def _validate_thresholds(
    metrics: MetricsSpec,
    performance: PerformanceSpec,
) -> None:
    if not math.isfinite(metrics.peak_threshold) or not 0 <= metrics.peak_threshold <= 1:
        raise ValueError("metrics.peak_threshold must be in [0, 1].")
    if (
        not math.isfinite(metrics.ball_distance_threshold)
        or metrics.ball_distance_threshold < 0
    ):
        raise ValueError("metrics.ball_distance_threshold must be finite and non-negative.")
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
        raise ValueError("performance.max_batches_per_split must be positive when set.")


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return cast(dict[str, Any], value)


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _integer(value, name="manifest.performance.max_batches_per_split")


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string.")
    result = value
    if not result.strip() or result != result.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean.")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if type(value) not in (float, int):
        raise TypeError(f"{name} must be a number.")
    return float(cast(float | int, value))


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
