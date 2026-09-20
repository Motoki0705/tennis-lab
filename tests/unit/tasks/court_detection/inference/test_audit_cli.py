"""Audit CLI paths are validated before inference or artifact publication."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from src.tasks.court_detection.geometry.hybrid_homography import DEFAULT_HYBRID_CONFIG
from src.tasks.court_detection.model_io.contracts import CourtLinePrediction
from src.tasks.court_detection.scripts import audit_hybrid_inference as audit
from src.utils.configuration import PathContractError
from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
    geometry_prediction,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"mock checkpoint")
    photo = tmp_path / "photo.png"
    Image.new("RGB", (12, 8)).save(photo)
    return checkpoint, photo, tmp_path / "output" / "audit"


@pytest.mark.parametrize("invalid", ["checkpoint", "image", "scene_root", "duplicate"])
def test_invalid_paths_fail_before_creating_output_or_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    checkpoint, photo, output = _inputs(tmp_path)
    if invalid == "checkpoint":
        checkpoint = tmp_path  # Existing directory is not a checkpoint file.
    if invalid == "image":
        photo = tmp_path / "missing.png"
    arguments = [
        "audit",
        "--checkpoint",
        str(checkpoint),
        "--image",
        str(photo),
        "--output-dir",
        str(output),
    ]
    if invalid == "scene_root":
        arguments += ["--scene-root", str(photo)]  # Existing file is not a directory.
    if invalid == "duplicate":
        arguments += ["--image", str(photo)]
    monkeypatch.setattr(sys, "argv", arguments)
    load = Mock()
    monkeypatch.setattr(audit.CourtPredictor, "load_from_checkpoint", load)
    with pytest.raises(PathContractError):
        audit.main()
    load.assert_not_called()
    assert not output.parent.exists()


@pytest.mark.parametrize("include_scene", [False, True])
def test_valid_paths_preserve_raw_evidence_and_optional_scene_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_scene: bool
) -> None:
    checkpoint, photo, output = _inputs(tmp_path)
    arguments = [
        "audit",
        "--checkpoint",
        str(checkpoint),
        "--image",
        str(photo),
        "--output-dir",
        str(output),
    ]
    scene_root = tmp_path / "scenes"
    owner = scene_root / "B00/alignment/alignment.json"
    if include_scene:
        owner.parent.mkdir(parents=True)
        owner.write_bytes(b'{"checkpoint":"historical"}')
        arguments += ["--scene-root", str(scene_root)]
    monkeypatch.setattr(sys, "argv", arguments)
    raw = geometry_prediction()
    probability = torch.linspace(0, 1, 24).reshape(4, 6)
    prediction = replace(
        raw,
        raw_heads={
            **raw.raw_heads,
            "line": CourtLinePrediction(
                probability=probability, logits=torch.zeros(4, 6)
            ),
        },
    )
    predict = Mock(return_value=prediction)
    predictor = SimpleNamespace(
        predict=predict,
        checkpoint_identity={"schema": "test"},
        hybrid_config=DEFAULT_HYBRID_CONFIG,
    )
    load = Mock(return_value=predictor)
    monkeypatch.setattr(audit.CourtPredictor, "load_from_checkpoint", load)
    audit.main()
    load.assert_called_once_with(checkpoint, device="cpu")
    assert predict.call_count == 1
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["scene_owners_unchanged"] is True
    assert len(metrics["scene_owner_sha256_before_and_after"]) == int(include_scene)
    with np.load(output / metrics["images"][0]["predictions"]) as saved:
        np.testing.assert_array_equal(saved["line_probability"], probability.numpy())
        assert saved["fitted_valid"].sum() == 12
        assert saved["selected"].sum() == 4
    if include_scene:
        assert owner.read_bytes() == b'{"checkpoint":"historical"}'
