"""Preserve eager pose validation while avoiding a second full dataset scan."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.tasks.court_detection.data import mixed as module
from src.tasks.court_detection.training.runner_mixed import (
    resolve_mixed_training_config,
)


def build(
    monkeypatch: pytest.MonkeyPatch, *, corrupt: bool = False
) -> tuple[module.MixedCourtDetectionDataModule, list[tuple[str, bool]]]:
    recipe = (
        Path(__file__).resolve().parents[5]
        / "scripts/colab/train/court_vit_ablation/baseline.yaml"
    )
    config, mixed = resolve_mixed_training_config(OmegaConf.load(recipe))
    calls: list[tuple[str, bool]] = []

    def factory(data: Any, *, is_train: bool, require_pose: bool) -> Any:
        def preflight(records: tuple[object, ...]) -> None:
            assert records
            calls.append((data.source.kind, is_train))
            if corrupt and require_pose:
                raise ValueError("corrupted pose authority")

        return SimpleNamespace(
            input_layer=SimpleNamespace(
                available_splits=("train", "val", "test"),
                records=lambda split: (object(),),
            ),
            target_bundle_spec=SimpleNamespace(kinds=()),
            preflight=preflight,
        )

    monkeypatch.setattr(module, "build_court_processing_pipeline", factory)
    return module.MixedCourtDetectionDataModule(config, mixed_config=mixed), calls


def test_setup_reuses_the_datasets_validated_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, calls = build(monkeypatch)
    assert len(calls) == 3
    assert all(name == "synthetic_court" for name, _ in calls)
    data.setup("fit")
    assert data.train_dataset.datasets[0] is data._validated_synthetic_datasets["train"]  # type: ignore[union-attr]
    assert data.val_dataset.datasets[0] is data._validated_synthetic_datasets["val"]  # type: ignore[union-attr]
    data.setup("test")
    assert sum(name == "synthetic_court" for name, _ in calls) == 3
    assert sum(name == "tennis_court_detector" for name, _ in calls) == 3


def test_invalid_pose_still_fails_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="corrupted pose authority"):
        build(monkeypatch, corrupt=True)
