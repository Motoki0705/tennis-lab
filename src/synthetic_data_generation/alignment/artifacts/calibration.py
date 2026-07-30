"""Publish and load fit-side alignment metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from src.synthetic_data_generation.alignment.artifacts.common import (
    load_json_artifact,
    publish_json_artifact,
    validate_artifact_id,
)

ALIGNMENT_CALIBRATION_SCHEMA = "court_alignment_calibration_v1"


def publish_calibration_artifact(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    """Publish one fit-side alignment metrics artifact."""
    return cast(
        Path,
        publish_json_artifact(
            payload,
            output_dir=output_dir,
            validate=_validate_calibration_payload,
            artifact_type="calibration",
        ),
    )


def load_calibration_artifact(path: Path) -> dict[str, Any]:
    """Load one fit-side alignment metrics artifact."""
    return cast(
        dict[str, Any],
        load_json_artifact(
            path,
            validate=_validate_calibration_payload,
            artifact_type="calibration",
        ),
    )


def _validate_calibration_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != ALIGNMENT_CALIBRATION_SCHEMA:
        raise ValueError(
            f"Unsupported calibration artifact schema: {payload.get('schema')!r}."
        )
    validate_artifact_id(payload.get("artifact_id"), artifact_type="calibration")
    required = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "provider",
        "geometry",
        "split",
        "detector",
        "evaluation_settings",
        "thresholds",
        "metrics",
        "threshold_comparisons",
        "stability",
        "point_cloud_support",
        "status",
        "provenance",
    }
    optional = {"artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Calibration artifact keys do not match its schema.")
    split = payload.get("split")
    if (
        not isinstance(split, dict)
        or split.get("holdout_inference_status") != "not_run"
    ):
        raise ValueError("Calibration must not infer holdout images.")
    if payload.get("status") != "metrics_recorded":
        raise ValueError("Invalid calibration status.")
    comparisons = payload.get("threshold_comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        raise ValueError("Threshold comparisons must be a non-empty mapping.")
    if not all(isinstance(value, bool) for value in comparisons.values()):
        raise ValueError("Threshold comparisons must contain booleans only.")
