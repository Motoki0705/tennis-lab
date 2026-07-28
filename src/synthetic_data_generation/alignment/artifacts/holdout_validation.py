"""Publish and strictly load immutable holdout validation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.synthetic_data_generation.alignment.artifacts.common import (
    load_json_artifact,
    publish_json_artifact,
    validate_artifact_id,
)

ALIGNMENT_VALIDATION_SCHEMA = "court_alignment_holdout_validation_v1"


def publish_holdout_validation_artifact(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    """Publish one strict fingerprinted holdout validation artifact."""
    return publish_json_artifact(
        payload,
        output_dir=output_dir,
        validate=_validate_holdout_validation_payload,
        artifact_type="holdout-validation",
    )


def load_holdout_validation_artifact(path: Path) -> dict[str, Any]:
    """Load and fingerprint-verify one holdout validation artifact."""
    return load_json_artifact(
        path,
        validate=_validate_holdout_validation_payload,
        artifact_type="holdout-validation",
    )


def _validate_holdout_validation_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != ALIGNMENT_VALIDATION_SCHEMA:
        raise ValueError(
            "Unsupported holdout validation artifact schema: "
            f"{payload.get('schema')!r}."
        )
    validate_artifact_id(
        payload.get("artifact_id"),
        artifact_type="holdout-validation",
    )
    required = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "provider",
        "geometry",
        "calibration",
        "split",
        "detector",
        "evaluation_settings",
        "gates",
        "metrics",
        "gate_results",
        "records",
        "status",
        "provenance",
    }
    optional = {"artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Holdout validation artifact keys do not match its schema.")
    split = payload.get("split")
    if (
        not isinstance(split, dict)
        or split.get("holdout_inference_status") != "complete"
    ):
        raise ValueError("Validation must record completed holdout inference.")
    if payload.get("status") not in {"accepted", "rejected"}:
        raise ValueError("Invalid holdout validation status.")
    gate_results = payload.get("gate_results")
    if not isinstance(gate_results, dict) or not gate_results:
        raise ValueError("Holdout gate_results must be a non-empty mapping.")
    if not all(isinstance(value, bool) for value in gate_results.values()):
        raise ValueError("Holdout gate_results must contain booleans only.")
