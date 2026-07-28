"""Publish and strictly load immutable fit-side calibration artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    """Publish one strict fingerprinted fit calibration artifact."""
    return publish_json_artifact(
        payload,
        output_dir=output_dir,
        validate=_validate_calibration_payload,
        artifact_type="calibration",
    )


def load_calibration_artifact(path: Path) -> dict[str, Any]:
    """Load and fingerprint-verify one fit calibration artifact."""
    return load_json_artifact(
        path,
        validate=_validate_calibration_payload,
        artifact_type="calibration",
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
        "gates",
        "metrics",
        "gate_results",
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
    if payload.get("status") not in {
        "fit_calibration_passed",
        "fit_calibration_failed",
    }:
        raise ValueError("Invalid calibration status.")
    gate_results = payload.get("gate_results")
    if not isinstance(gate_results, dict) or not gate_results:
        raise ValueError("Calibration gate_results must be a non-empty mapping.")
    if not all(isinstance(value, bool) for value in gate_results.values()):
        raise ValueError("Calibration gate_results must contain booleans only.")
