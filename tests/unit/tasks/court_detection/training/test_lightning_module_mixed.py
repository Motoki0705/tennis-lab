"""Tests for mixed Court run-specific artifact isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.court_detection.training.lightning_module_mixed import (
    MixedCourtDetectionLightningModule,
)
from src.utils.configuration import PathResolver, RuntimePathRoots

pytestmark = pytest.mark.unit


def _module(
    tmp_path: Path,
    *,
    output_key: str,
) -> MixedCourtDetectionLightningModule:
    module = object.__new__(MixedCourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    module.path_resolver = PathResolver(
        RuntimePathRoots(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            checkpoint_root=tmp_path / "checkpoints",
            artifact_root=tmp_path / "artifacts",
            output_root=tmp_path / "outputs",
            cache_root=tmp_path / "cache",
            external_asset_root=tmp_path / "external",
        )
    )
    module._test_prediction_output_key = Path(output_key)
    return module


def test_non_queue_predictions_are_isolated_by_variant_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    dense = _module(
        tmp_path,
        output_key="court_detection/mixed-source/dense-only",
    )
    pose = _module(
        tmp_path,
        output_key="court_detection/mixed-source/dense-pose",
    )

    dense_dir = dense._test_predictions_dir()
    pose_dir = pose._test_predictions_dir()

    assert dense_dir == (
        tmp_path / "artifacts/test_predictions/court_detection/mixed-source/dense-only"
    )
    assert pose_dir == (
        tmp_path / "artifacts/test_predictions/court_detection/mixed-source/dense-pose"
    )
    assert dense_dir != pose_dir


def test_queue_predictions_keep_repro_bundle_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(
        tmp_path,
        output_key="court_detection/mixed-source/dense-only",
    )
    repro_dir = tmp_path / "queue-repro"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))

    assert module._test_predictions_dir() == repro_dir / "predictions"
