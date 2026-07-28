"""Tests for immutable holdout validation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.synthetic_data_generation.alignment.artifacts.holdout_validation import (
    ALIGNMENT_VALIDATION_SCHEMA,
    load_holdout_validation_artifact,
    publish_holdout_validation_artifact,
)


def _payload() -> dict[str, Any]:
    return {
        "schema": ALIGNMENT_VALIDATION_SCHEMA,
        "artifact_id": "holdout-v1",
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "provider": {},
        "geometry": {},
        "calibration": {},
        "split": {"holdout_inference_status": "complete"},
        "detector": {},
        "evaluation_settings": {},
        "gates": {},
        "metrics": {},
        "gate_results": {"holdout_gate": False},
        "records": [],
        "status": "rejected",
        "provenance": {},
    }


def test_holdout_validation_round_trip_preserves_rejection(tmp_path: Path) -> None:
    path = publish_holdout_validation_artifact(_payload(), output_dir=tmp_path)

    loaded = load_holdout_validation_artifact(path)

    assert loaded["status"] == "rejected"
    assert loaded["gate_results"] == {"holdout_gate": False}


def test_holdout_validation_rejects_calibration_schema(tmp_path: Path) -> None:
    payload = _payload()
    payload["schema"] = "court_alignment_calibration_v1"

    with pytest.raises(ValueError, match="Unsupported holdout validation"):
        publish_holdout_validation_artifact(payload, output_dir=tmp_path)
