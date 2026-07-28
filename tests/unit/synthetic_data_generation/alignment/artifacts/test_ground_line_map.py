"""Tests for immutable ground-line map artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.artifacts.ground_line_map import (
    GROUND_LINE_MAP_SCHEMA,
    load_ground_line_map_artifact,
    publish_ground_line_map_artifact,
)


def _arrays() -> dict[str, np.ndarray]:
    evidence: np.ndarray = np.zeros((3, 4), dtype=np.float32)
    evidence[1, 2] = 0.5
    weights: np.ndarray = np.zeros_like(evidence)
    weights[1, 2] = 1.0
    support: np.ndarray = np.zeros(evidence.shape, dtype=np.uint16)
    support[1, 2] = 1
    return {
        "evidence_sum": evidence,
        "weight_sum": weights,
        "view_count": support,
        "mean_probability": evidence.copy(),
    }


def _payload() -> dict[str, object]:
    return {
        "schema": GROUND_LINE_MAP_SCHEMA,
        "artifact_id": "synthetic-ground-lines-v1",
        "created_at_utc": "2026-07-25T00:00:00+00:00",
        "provider": {},
        "split": {
            "fit_camera_ids": ["fit-0"],
            "holdout_camera_ids": ["holdout-0"],
            "holdout_inference_status": "not_run",
        },
        "detector": {},
        "ground_plane": {},
        "projection": {},
        "records": [{"camera_id": "fit-0"}],
        "summary": {},
        "provenance": {},
    }


def test_artifact_round_trip_refuses_overwrite_and_detects_tampering(
    tmp_path: Path,
) -> None:
    payload = _payload()
    arrays = _arrays()
    path = publish_ground_line_map_artifact(
        payload,
        arrays=arrays,
        output_dir=tmp_path,
    )
    manifest, loaded_arrays = load_ground_line_map_artifact(path)

    assert manifest["split"]["holdout_inference_status"] == "not_run"
    assert loaded_arrays["evidence_sum"].shape == arrays["evidence_sum"].shape
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_ground_line_map_artifact(
            payload,
            arrays=arrays,
            output_dir=tmp_path,
        )

    manifest_path = path / "manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["summary"]["changed"] = True
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_ground_line_map_artifact(path)


def test_artifact_rejects_inconsistent_mean_probability(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays["mean_probability"][1, 2] = 0.25

    with pytest.raises(ValueError, match="mean_probability"):
        publish_ground_line_map_artifact(
            _payload(),
            arrays=arrays,
            output_dir=tmp_path,
        )
