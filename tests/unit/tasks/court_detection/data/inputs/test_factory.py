"""Contract tests for the Court input factory's split selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.tasks.court_detection.configuration import (
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)

pytestmark = pytest.mark.unit


def _write_tennis_source(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    points = [[float(index + 1), float(index + 2)] for index in range(14)]
    for sample_id in ("train-1", "val-1"):
        Image.new("RGB", (32, 24)).save(root / "images" / f"{sample_id}.png")
    (root / "data_train.json").write_text(
        json.dumps([{"id": "train-1", "kps": points}]), encoding="utf-8"
    )
    (root / "data_val.json").write_text(
        json.dumps([{"id": "val-1", "kps": points}]), encoding="utf-8"
    )


def _tennis_config(root: Path) -> TennisCourtDetectorSourceConfig:
    return TennisCourtDetectorSourceConfig(
        kind="tennis_court_detector",
        root=root,
        split_mapping={"train": "train", "val": "val", "test": None},
        excluded_sample_ids=(),
    )


def test_the_factory_reads_every_configured_split_by_default(tmp_path: Path) -> None:
    _write_tennis_source(tmp_path / "court")

    input_layer = build_court_input(
        _tennis_config(tmp_path / "court"),
        target_store=CourtDerivedTargetStore(tmp_path / "derived"),
    )

    assert input_layer.available_splits == ("train", "val")


def test_the_factory_forwards_requested_splits_to_the_tennis_input(
    tmp_path: Path,
) -> None:
    _write_tennis_source(tmp_path / "court")

    input_layer = build_court_input(
        _tennis_config(tmp_path / "court"),
        target_store=CourtDerivedTargetStore(tmp_path / "derived"),
        requested_splits=("val",),
    )

    assert input_layer.available_splits == ("val",)


def test_the_synthetic_input_rejects_requested_splits(tmp_path: Path) -> None:
    """A source without a partial read must say so instead of ignoring the request."""
    config = SyntheticCourtSourceConfig(
        kind="synthetic_court",
        schema="v3",
        court_scope="target_court",
        workspace_root=tmp_path / "scenes",
        scene_ids=("B00",),
    )

    with pytest.raises(ValueError, match="no partial split read"):
        build_court_input(
            config,
            target_store=CourtDerivedTargetStore(tmp_path / "derived"),
            requested_splits=("test",),
        )
