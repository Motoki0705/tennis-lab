"""Tests for immutable fit-side calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.synthetic_data_generation.alignment.artifacts.calibration import (
    ALIGNMENT_CALIBRATION_SCHEMA,
    load_calibration_artifact,
    publish_calibration_artifact,
)


def _payload() -> dict[str, Any]:
    return {
        "schema": ALIGNMENT_CALIBRATION_SCHEMA,
        "artifact_id": "calibration-v1",
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "provider": {},
        "geometry": {},
        "split": {"holdout_inference_status": "not_run"},
        "detector": {},
        "evaluation_settings": {},
        "gates": {},
        "metrics": {},
        "gate_results": {"fit_gate": True},
        "stability": {},
        "point_cloud_support": {},
        "status": "fit_calibration_passed",
        "provenance": {},
    }


def test_calibration_artifact_round_trip_detects_tampering(tmp_path: Path) -> None:
    payload = _payload()
    path = publish_calibration_artifact(payload, output_dir=tmp_path)

    assert load_calibration_artifact(path)["status"] == "fit_calibration_passed"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_calibration_artifact(payload, output_dir=tmp_path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["metrics"]["tampered"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_calibration_artifact(path)


def test_calibration_rejects_holdout_inference(tmp_path: Path) -> None:
    payload = _payload()
    payload["split"]["holdout_inference_status"] = "complete"

    with pytest.raises(ValueError, match="must not infer holdout"):
        publish_calibration_artifact(payload, output_dir=tmp_path)
