"""Publish and strictly load immutable metric court-geometry artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.synthetic_data_generation.alignment.artifacts.common import (
    load_json_artifact,
    publish_json_artifact,
    validate_artifact_id,
)

COURT_GEOMETRY_SCHEMA = "ground_court_geometry_v1"


def publish_court_geometry_artifact(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    """Publish one immutable fingerprinted court-geometry JSON artifact."""
    return publish_json_artifact(
        payload,
        output_dir=output_dir,
        validate=_validate_payload,
        artifact_type="court-geometry",
    )


def load_court_geometry_artifact(path: Path) -> dict[str, Any]:
    """Load and fingerprint-verify a court-geometry artifact."""
    return load_json_artifact(
        path,
        validate=_validate_payload,
        artifact_type="court-geometry",
    )


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != COURT_GEOMETRY_SCHEMA:
        raise ValueError(
            f"Unsupported court-geometry schema: {payload.get('schema')!r}."
        )
    validate_artifact_id(payload.get("artifact_id"), artifact_type="court-geometry")
    required = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "ground_line_artifact",
        "fit_settings",
        "candidates",
        "selection",
        "acceptance_status",
        "provenance",
    }
    optional = {"artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Court-geometry manifest keys do not match v1 schema.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Court-geometry candidates must be a non-empty list.")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("Every court-geometry candidate must be an object.")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Court-geometry selection must be an object.")
    candidate_ids = {candidate.get("candidate_id") for candidate in candidates}
    if selection.get("selected_candidate_id") not in candidate_ids:
        raise ValueError("Selected court candidate does not exist.")
    if payload.get("acceptance_status") != "fit_candidate_holdout_not_run":
        raise ValueError("Court geometry must remain a fit candidate before holdout.")
